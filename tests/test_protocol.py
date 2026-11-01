"""Synthetic verification, never exam metric results."""
import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import pipeline as p
import numpy as np
import pandas as pd

class Protocol(unittest.TestCase):
    def test_interval_against_independent_bruteforce(self):
        rng=np.random.default_rng(41)
        for _ in range(25):
            starts=np.sort(rng.integers(0,100,40)); ends=starts+rng.integers(1,60,40)
            ss=rng.integers(0,120,30); se=ss+rng.integers(1,70,30)
            si,ai=p.interval_pairs(starts,ends,ss,se)
            actual=set(zip(si.tolist(),ai.tolist()))
            expected={(i,j) for i in range(len(ss)) for j in range(len(starts)) if min(se[i],ends[j])-max(ss[i],starts[j])>0}
            self.assertTrue(expected); self.assertEqual(actual,expected)
        si,ai=p.interval_pairs(np.array([0,5,10]),np.array([100,7,20]),np.array([8]),np.array([15]))
        self.assertEqual(set(ai),{0,2})

    def test_union_and_threshold(self):
        d=pd.DataFrame([['0001',1,0,60],['0001',1,30,90],['0001',1,30,90],['0001',1,95,100],['0002',1,10,20]],columns=['u','a','s','e'])
        v=p.union_fragments(d).set_index('u').seconds
        self.assertEqual(v['0001'],95); self.assertEqual(v['0002'],10)
        self.assertTrue(v['0001']/100>=.8)
        self.assertFalse(79/100>=.8)

    def test_dates_ids_boundaries(self):
        self.assertEqual(p.seconds(pd.Series(['09.03.2020 06:00:00']),'%d.%m.%Y %H:%M:%S')[0],p.epoch('2020-03-09 06:00:00'))
        from io import StringIO
        d=pd.read_csv(StringIO('user_id,item_id\n18446744073709551615,000123\n'),dtype=str)
        self.assertEqual(d.user_id[0],'18446744073709551615'); self.assertEqual(d.item_id[0],'000123')
        union=pd.DataFrame({'u':['x']*3,'tv_show_id':['a']*3,'s':[p.epoch(p.CUT)-100,p.epoch(p.CUT)-50,p.epoch(p.CUT)],'e':[p.epoch(p.CUT),p.epoch(p.CUT)+50,p.epoch(p.CUT)+100],'seconds':[100]*3,'duration_s':[100]*3})
        train,_=p.interaction_matrix(union,['x'],['a'],p.START,p.CUT)
        val,_=p.interaction_matrix(union,['x'],['a'],p.CUT,p.END)
        self.assertEqual(train[0,0],1); self.assertEqual(val[0,0],1)

    def test_ap_independent_arithmetic(self):
        # Two of three relevant at ranks 1,3: (1 + 2/3)/3 = 5/9.
        self.assertAlmostEqual(p.ap5(['a','x','b','z','w'],['a','b','c']),5/9)
        self.assertEqual(p.ap5(['x'],['a']),0)
        self.assertEqual(p.ap5(['a'],[]),0)
        self.assertEqual(p.ap5(['a'],['a']),1)
        self.assertAlmostEqual(p.ap5(['a','a','b'],['a','b']), (1+2/3)/2)
        self.assertNotEqual(p.ap5(['a','x','b'],['a','b']),p.ap5(['x','a','b'],['a','b']))

    def test_tie_fallback(self):
        pop=np.array([1,4,4,0,3,2.]); eligible=np.arange(6)
        self.assertEqual(list(p.top5(np.zeros(6),eligible,pop)),[1,2,4,5,0])
        self.assertEqual(list(p.top5(np.array([2,0,0,0,0,0]),eligible,pop))[0],0)

    def test_als_unobserved_items_zero(self):
        from scipy import sparse
        u,v=p.fit_als(sparse.csr_matrix([[2.,0,1],[0,0,1]]),factors=3,iterations=2)
        self.assertTrue(np.all(v[1]==0)); self.assertTrue(np.isfinite(u).all())
        compact=p.sparse.csr_matrix([[2.,1],[0,1]])
        u2,v2=p.fit_als(compact,factors=3,iterations=2)
        np.testing.assert_array_equal(u,u2)
        np.testing.assert_array_equal(v[[0,2]],v2)

if __name__=='__main__': unittest.main(verbosity=2)
