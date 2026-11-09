"""Unit tests for Phase D Focal Tversky Loss implementation."""

import unittest
import torch
import torch.nn.functional as F

def focal_tversky_loss(logits, target, alpha=0.7, beta=0.3, gamma=4/3, eps=1e-5):
    logits, target = logits.float(), target.float()
    p = logits.sigmoid()
    axes = tuple(range(1, logits.ndim))
    tp = (p * target).sum(axes)
    fp = (p * (1.0 - target)).sum(axes)
    fn = ((1.0 - p) * target).sum(axes)
    ti = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    ti = torch.clamp(ti, 0.0, 1.0)
    loss = torch.pow(1.0 - ti, gamma)
    return loss.mean()

class FocalTverskyLossTests(unittest.TestCase):
    def test_perfect_prediction_loss_near_zero(self):
        # Target has foreground
        target = torch.zeros((2, 1, 16, 16, 16))
        target[:, :, 4:12, 4:12, 4:12] = 1.0
        # Logits strongly positive where target is 1, strongly negative where 0
        logits = torch.full_like(target, -10.0)
        logits[target == 1.0] = 10.0
        
        loss = focal_tversky_loss(logits, target, alpha=0.7, beta=0.3, gamma=4/3)
        self.assertTrue(torch.isfinite(loss))
        self.assertLess(loss.item(), 1e-4)

    def test_completely_wrong_prediction_loss_near_one(self):
        target = torch.zeros((1, 1, 16, 16, 16))
        target[:, :, 4:12, 4:12, 4:12] = 1.0
        # Opposite prediction
        logits = torch.full_like(target, 10.0)
        logits[target == 1.0] = -10.0
        
        loss = focal_tversky_loss(logits, target, alpha=0.7, beta=0.3, gamma=4/3)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.95)

    def test_empty_target_numerical_stability(self):
        # Empty target (all 0s)
        target = torch.zeros((2, 1, 16, 16, 16))
        logits = torch.randn((2, 1, 16, 16, 16))
        loss = focal_tversky_loss(logits, target, alpha=0.7, beta=0.3, gamma=4/3)
        self.assertTrue(torch.isfinite(loss))

    def test_gradients_propagate_cleanly(self):
        target = torch.zeros((1, 1, 8, 8, 8))
        target[:, :, 2:6, 2:6, 2:6] = 1.0
        logits = torch.zeros((1, 1, 8, 8, 8), requires_grad=True)
        
        loss = focal_tversky_loss(logits, target, alpha=0.7, beta=0.3, gamma=4/3)
        loss.backward()
        
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertNotEqual(logits.grad.abs().sum().item(), 0.0)

    def test_alpha_penalizes_false_positives_more_heavily(self):
        # Base setup: target has a 4x4x4 block
        target = torch.zeros((1, 1, 10, 10, 10))
        target[:, :, 3:7, 3:7, 3:7] = 1.0
        
        # Case with FP: model predicts the true block AND an extra 4x4x4 block elsewhere
        logits_fp = torch.full_like(target, -5.0)
        logits_fp[:, :, 3:7, 3:7, 3:7] = 5.0 # TP
        logits_fp[:, :, 0:3, 0:3, 0:3] = 5.0 # FP
        
        # With alpha=0.7 (high FP penalty), loss should be strictly higher than with alpha=0.2
        loss_high_alpha = focal_tversky_loss(logits_fp, target, alpha=0.7, beta=0.3, gamma=1.0)
        loss_low_alpha = focal_tversky_loss(logits_fp, target, alpha=0.2, beta=0.8, gamma=1.0)
        
        self.assertGreater(loss_high_alpha.item(), loss_low_alpha.item())

if __name__ == "__main__":
    unittest.main()

