"""ARU-Net baseline detection model wrapper.

Loads the pre-trained TensorFlow frozen graph from the original ARU-Net repo
and exposes it via a simple forward() interface compatible with our ensemble.

ARU-Net predicts baseline and separator pixel locations for historical documents.
Output: [H, W] with 2 channels:
  - Channel 0: baseline probability
  - Channel 1: separator probability

For our pipeline, we use channel 0 (baseline).
"""
import numpy as np
import torch
import torch.nn as nn

try:
    import tensorflow as tf
except ImportError:
    raise ImportError("ARU-Net requires TensorFlow. Install: pip install tensorflow")


class ARUNetWrapper(nn.Module):
    """
    Wraps TensorFlow frozen graph inference in a PyTorch-compatible interface.

    Args:
        model_path: path to .pb frozen graph file (e.g., demo_nets/model100_ema.pb)
        device: torch device (used only for interface compatibility; inference is on TensorFlow)
    """

    def __init__(self, model_path: str, device: str = "cpu", scale: float = 0.33):
        super().__init__()
        self.device_str = device
        self.scale = scale  # Image scale factor (default 0.33 like original ARU-Net)

        # Load frozen TensorFlow graph (ARU-Net format)
        with tf.io.gfile.GFile(model_path, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.import_graph_def(graph_def, name="")

        # Configure session to use CPU by default (less memory hungry)
        sess_config = tf.compat.v1.ConfigProto()
        if device == "cpu":
            sess_config.device_count["GPU"] = 0
        self.sess = tf.compat.v1.Session(graph=self.graph, config=sess_config)

        # ARU-Net frozen graph uses these tensor names
        self.input_tensor = self.graph.get_tensor_by_name("inImg:0")
        self.output_tensor = self.graph.get_tensor_by_name("output:0")

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Run inference on image(s).

        Args:
            img: [B, H, W, 3] or [H, W, 3] tensor (uint8 or float32, RGB or grayscale)

        Returns:
            [B, H, W, 2] or [H, W, 2] tensor with baseline and separator predictions.
            Shape convention matches TensorFlow (batch, height, width, channels).
        """
        # Handle batch vs single image
        if img.ndim == 3:
            img = img.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Convert to numpy
        img_np = img.cpu().numpy()

        # Convert to uint8 (ARU-Net expects uint8, not normalized floats)
        if img_np.dtype == np.float32:
            if img_np.max() <= 1.5:
                img_np = (img_np * 255.0).astype(np.uint8)
            else:
                img_np = np.clip(img_np, 0, 255).astype(np.uint8)

        # Convert RGB to grayscale if needed
        if img_np.shape[-1] == 3:
            # RGB -> grayscale: 0.299*R + 0.587*G + 0.114*B (standard formula)
            img_np = (
                0.299 * img_np[..., 0].astype(np.float32)
                + 0.587 * img_np[..., 1].astype(np.float32)
                + 0.114 * img_np[..., 2].astype(np.float32)
            ).astype(np.uint8)
            # Add channel dimension: [B, H, W] -> [B, H, W, 1]
            img_np = np.expand_dims(img_np, axis=-1)
        elif img_np.shape[-1] == 1:
            # Already [B, H, W, 1], ensure uint8
            img_np = img_np.astype(np.uint8)
        else:
            raise ValueError(f"Unexpected image shape: {img_np.shape}")

        # Scale down image if requested (reduces memory usage)
        original_shape = img_np.shape
        if self.scale < 1.0:
            import cv2
            scaled_imgs = []
            for i in range(img_np.shape[0]):
                h, w = img_np[i].shape[:2]
                new_h, new_w = int(h * self.scale), int(w * self.scale)
                scaled = cv2.resize(img_np[i, :, :, 0], (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                scaled_imgs.append(np.expand_dims(scaled, axis=-1))
            img_np = np.stack(scaled_imgs, axis=0)

        # Run TensorFlow inference
        with self.graph.as_default():
            output_np = self.sess.run(
                self.output_tensor,
                feed_dict={self.input_tensor: img_np},
            )

        # Resize output back to original size if we scaled
        if self.scale < 1.0:
            import cv2
            h_orig, w_orig = original_shape[1], original_shape[2]
            output_resized = []
            for i in range(output_np.shape[0]):
                # Resize each channel back to original resolution
                channels = []
                for c in range(output_np.shape[3]):
                    ch = cv2.resize(output_np[i, :, :, c], (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
                    channels.append(ch)
                output_resized.append(np.stack(channels, axis=-1))
            output_np = np.stack(output_resized, axis=0)
            # Note: output_np is already in [0, 1] range from TensorFlow

        # Convert back to torch tensor
        output_tensor = torch.from_numpy(output_np).float()

        if squeeze_output:
            output_tensor = output_tensor.squeeze(0)

        return output_tensor

    def predict(self, img: torch.Tensor) -> torch.Tensor:
        """
        Alias for forward(). Returns sigmoid of output logits.
        """
        logits = self.forward(img)
        return torch.sigmoid(logits)

    def __del__(self):
        """Clean up TensorFlow session."""
        if hasattr(self, "sess"):
            self.sess.close()
