import pytest
from lpr_tools.eval.detector_size_study import stratified_sample,metrics,paired_changes,summary


def test_sampling_exact_proportional_reproducible_and_original_order():
    rows=[{'path':f'{s}/{i}','subset':s} for s,n in [('base',500),('green',500),*[(f'h{k}',300) for k in range(7)]] for i in range(n)]
    sample=stratified_sample(rows)
    from collections import Counter
    assert Counter(r['subset'] for r in sample)=={'base':97,'green':97,**{f'h{k}':58 for k in range(7)}}
    assert sample==stratified_sample(rows)
    assert sample!=stratified_sample(rows,seed=43)
    assert [rows.index(r) for r in sample]==sorted(rows.index(r) for r in sample)
    with pytest.raises(ValueError):stratified_sample(rows+rows[:1])


def test_metrics_denominators_and_paired_changes():
    rows=[{'path':'a','truth':'A','plate':'A','status':'LOW_CONFIDENCE','subset':'x'},
          {'path':'b','truth':'B','plate':'wrong','status':'OK','subset':'x'},
          {'path':'c','truth':'C','plate':None,'status':'NO_PLATE','subset':'x'}]
    result=metrics(rows)
    assert result['correct']==1 and result['accuracy']==1/3
    assert result['auto_accepted']==1 and result['auto_precision']==0 and result['no_plate']==1
    assert metrics([])['auto_precision'] is None
    paired=paired_changes(rows,[{'path':'a','plate':'wrong'},{'path':'b','plate':'B'},{'path':'c','plate':None}])
    assert paired['improved']==paired['regressed']==1
    assert paired['representatives']['improved'][0]['path']=='a'
    with pytest.raises(ValueError):paired_changes(rows,[])


def test_native_composition_keeps_frozen_weights_and_dark_filter():
    subsets=['ccpd_base','ccpd_green','ccpd_db','ccpd_tilt','ccpd_rotate','ccpd_weather']
    rows=[{'subset':s,'partition':'val' if i<2 else 'devhard','truth':'A','plate':'A' if i%2==0 else 'B',
           'status':'OK','brightness':19} for i,s in enumerate(subsets)]
    rows += [{**rows[2],'brightness':20,'plate':'wrong'}]
    result=summary(rows)
    assert result['composition_estimate']=={'clear':.8,'night':1.,'tilt':.3,'dirty':0.}
    assert result['dark']['n']==1