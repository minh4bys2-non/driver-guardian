import math
import unittest

import torch

from src.model import NMSFreeDetector


class ReplaceHeadTests(unittest.TestCase):
    def assert_classifier(self, model, nc):
        self.assertEqual(model.nc, nc)
        self.assertEqual(model.head.nc, nc)
        for scale, stride in zip(model.head.heads, model.strides):
            self.assertEqual(scale.nc, nc)
            for branch in ('o2m', 'o2o'):
                layer = getattr(scale, f'out_cls_{branch}')
                self.assertEqual(layer.out_channels, nc)
                self.assertEqual(layer.weight.shape, (nc, layer.in_channels, 1, 1))
                bias = math.log(5 / nc / (model.img_size / stride) ** 2)
                torch.testing.assert_close(layer.bias, torch.full_like(layer.bias, bias))
                self.assertTrue(torch.isfinite(layer.weight).all())
                self.assertGreater(layer.weight.std().item(), 0)
        self.assertFalse(model.head.dfl.conv.weight.requires_grad)
        torch.testing.assert_close(model.head.dfl.conv.weight.flatten(),
                                   torch.arange(model.reg_max, device=next(model.parameters()).device,
                                                dtype=next(model.parameters()).dtype), rtol=0, atol=0)

    def test_partial_preserves_parameters_buffers_and_boxes(self):
        model = NMSFreeDetector(nc=4, img_size=64).double().eval()
        head = model.head
        params = dict(model.named_parameters())
        before = {k: v.clone() for k, v in model.state_dict().items()}
        x = torch.randn(2, 3, 64, 64, dtype=torch.float64)
        with torch.no_grad():
            original = model(x)
        self.assertIs(model.replace_head(nc=3, replace_all=False), model)
        self.assertIs(model.head, head)
        self.assert_classifier(model, 3)
        for name, value in model.state_dict().items():
            if 'out_cls_' not in name:
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        for name, param in model.named_parameters():
            if 'out_cls_' in name:
                self.assertIsNot(param, params[name])
            else:
                self.assertIs(param, params[name])
            self.assertEqual(param.dtype, torch.float64)
            self.assertEqual(param.device, params[name].device)
        self.assertTrue(all(not m.training for m in model.modules()))
        with torch.no_grad():
            output = model(x)
            o2o = model(x, o2o_only=True)
        for branch in ('o2m', 'o2o'):
            self.assertEqual(output[branch]['cls'].shape, (2, 84, 3))
            for key in ('reg_raw', 'box'):
                torch.testing.assert_close(output[branch][key], original[branch][key], rtol=0, atol=0)
        self.assertNotIn('o2m', o2o)
        torch.testing.assert_close(o2o['o2o']['cls'], output['o2o']['cls'])
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        model.train()
        output = model(x)
        sum(output[b]['cls'].sum() for b in ('o2m', 'o2o')).backward()
        weights = [(p, p.detach().clone()) for n, p in model.named_parameters() if 'out_cls_' in n]
        self.assertTrue(all(p.grad is not None for p, _ in weights))
        optimizer.step()
        self.assertTrue(all(not torch.equal(p, old) for p, old in weights))
        self.assertIsNone(model.head.dfl.conv.weight.grad)

    def test_partial_resets_same_nc_and_defaults(self):
        model = NMSFreeDetector(nc=3)
        for kwargs in ({'nc': 3}, {}, {'reg_max': model.reg_max, 'strides': list(model.strides),
                                     'img_size': model.img_size}):
            with self.subTest(kwargs=kwargs):
                old = [s.out_cls_o2m for s in model.head.heads]
                model.replace_head(replace_all=False, **kwargs)
                self.assert_classifier(model, 3)
                for scale, previous in zip(model.head.heads, old):
                    self.assertIsNot(scale.out_cls_o2m, previous)
                    self.assertFalse(torch.equal(scale.out_cls_o2m.weight, previous.weight))
                    self.assertTrue(scale.out_cls_o2m.training)

    def test_partial_rejects_invalid_arguments_without_mutation(self):
        model = NMSFreeDetector(nc=3)
        params = dict(model.named_parameters())
        before = {k: v.clone() for k, v in model.state_dict().items()}
        for kwargs in ({'nc': 0}, {'nc': -1}, {'nc': True}, {'nc': 1.5},
                       {'reg_max': 8}, {'strides': (4, 8, 16)}, {'img_size': 320}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                model.replace_head(replace_all=False, **kwargs)
            self.assert_classifier(model, 3)
            for name, p in model.named_parameters():
                self.assertIs(p, params[name])
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_full_default_and_explicit_replace_entire_head(self):
        for kwargs in ({}, {'replace_all': True}):
            with self.subTest(kwargs=kwargs):
                model = NMSFreeDetector(nc=4).double()
                old = model.head
                params = dict(model.named_parameters())
                trunk = {k: v.clone() for k, v in model.state_dict().items() if not k.startswith('head.')}
                self.assertIs(model.replace_head(nc=3, reg_max=8, strides=(4, 8, 16),
                                                 img_size=320, **kwargs), model)
                self.assertIsNot(model.head, old)
                self.assert_classifier(model, 3)
                for obj in (model, model.head):
                    self.assertEqual((obj.reg_max, obj.strides, obj.img_size), (8, (4, 8, 16), 320))
                for name, p in model.named_parameters():
                    self.assertEqual(p.dtype, torch.float64)
                    if name.startswith('head.'):
                        self.assertIsNot(p, params[name])
                    else:
                        self.assertIs(p, params[name])
                for name, value in trunk.items():
                    torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
                for scale in model.head.heads:
                    for branch in ('o2m', 'o2o'):
                        layer = getattr(scale, f'out_reg_{branch}')
                        self.assertEqual(layer.out_channels, 32)
                        torch.testing.assert_close(layer.bias, torch.ones_like(layer.bias))


if __name__ == '__main__':
    unittest.main()
