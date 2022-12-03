import numpy as np
import pytest

from toynn.core import Tensor


class TestTensor:
    def test_init(self):
        t = Tensor([1, 2, 3])
        assert (t.data == np.array([1, 2, 3])).all()
        assert (t.grad == np.array([0, 0, 0])).all()

    def test_add(self):
        t1 = Tensor([1, 2, 3])
        t2 = Tensor([4, 5, 6])
        t3 = t1 + t2
        assert (t3.data == np.array([5, 7, 9])).all()
        assert (t3.grad == np.array([0, 0, 0])).all()
        t3.backward()
        assert (t1.grad == np.array([1, 1, 1])).all()
        assert (t2.grad == np.array([1, 1, 1])).all()

    def test_sub(self):
        t1 = Tensor([1, 2, 3])
        t2 = Tensor([4, 5, 6])
        t3 = t1 - t2
        assert (t3.data == np.array([-3, -3, -3])).all()
        assert (t3.grad == np.array([0, 0, 0])).all()
        t3.backward()
        assert (t1.grad == np.array([1, 1, 1])).all()
        assert (t2.grad == np.array([-1, -1, -1])).all()


if __name__ == '__main__':
    pytest.main()
