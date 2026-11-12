import unittest
import numpy as np
from export_results import spans,match_events


class Contract(unittest.TestCase):
    def test_tail_is_not_filled_and_last_cell_is_clipped(self):
        self.assertTrue(np.allclose(spans(np.array([True,False,True]),.16,.42),[[0,.16],[.32,.42]]))
        self.assertTrue(np.allclose(spans(np.array([True]),.16,.3),[[0,.16]]))

    def test_argmax_tie_is_fake(self):
        pred=np.array([[1.,1.],[1.,2.],[2.,1.]]).argmax(1)
        self.assertEqual(pred.tolist(),[0,1,0])

    def test_split_and_merge_have_different_denominators(self):
        ov,_,matched=match_events(np.array([[0.,1.],[2.,3.]]),np.array([[0.,.4],[.6,2.4]]))
        self.assertEqual((ov>0).sum(1).tolist(),[2,1])
        self.assertEqual((ov>0).sum(0).tolist(),[1,2])
        self.assertEqual(matched,{})

    def test_matching_uses_iou_and_one_to_one(self):
        ov,iou,matched=match_events(np.array([[0.,1.],[1.,2.]]),np.array([[0.,2.]]))
        self.assertEqual(len(matched),1)
        self.assertTrue(np.allclose(iou,.5))


if __name__=='__main__': unittest.main()
