# S1 Local Flip Echo Probe

Claim: `S1-ECHO-LOCAL-FLIP-C01`
Predict: `P-S1-ECHO-LOCAL01`
Host: `io.grepcode.cn`

## Result

pilot_pass: `True`

```text
hard_local_treeheap_closure_exact = 1.000000
learned_ood_exact = 1.000000
learned_ood_token_acc = 1.000000
learned_ood_edit_similarity = 1.000000
no_inverse_ood_exact = 0.001000
```

## Readable Examples

- len=11 span=(0,4) exact=True
  - observed: `worse or better for, that simply is not true.`
  - restored: `for better or worse, that simply is not true.`
  - target: `for better or worse, that simply is not true.`
- len=10 span=(1,8) exact=True
  - observed: `many development positive a is this that argue would.`
  - restored: `many would argue that this is a positive development.`
  - target: `many would argue that this is a positive development.`
- len=13 span=(2,7) exact=True
  - observed: `the growth yet has, course of, crisis to be resolved.`
  - restored: `the growth crisis, of course, has yet to be resolved.`
  - target: `the growth crisis, of course, has yet to be resolved.`
- len=22 span=(1,8) exact=True
  - observed: `given the, crisis economic the of gravity the time has come for the eu to demonstrate its strengths whenever possible.`
  - restored: `given the gravity of the economic crisis, the time has come for the eu to demonstrate its strengths whenever possible.`
  - target: `given the gravity of the economic crisis, the time has come for the eu to demonstrate its strengths whenever possible.`
- len=12 span=(8,4) exact=True
  - observed: `fortunately, china s leaders seem intent on. that just doing`
  - restored: `fortunately, china s leaders seem intent on doing just that.`
  - target: `fortunately, china s leaders seem intent on doing just that.`
- len=24 span=(1,8) exact=True
  - observed: `this in impact disproportionate a having is trajectory upward africa, where the share of household income spent on food is also rising.`
  - restored: `this upward trajectory is having a disproportionate impact in africa, where the share of household income spent on food is also rising.`
  - target: `this upward trajectory is having a disproportionate impact in africa, where the share of household income spent on food is also rising.`
- len=24 span=(13,3) exact=True
  - observed: `moreover, though emerging economies are also experiencing acute growth slowdowns, their the of share global economic pie will continue to grow.`
  - restored: `moreover, though emerging economies are also experiencing acute growth slowdowns, their share of the global economic pie will continue to grow.`
  - target: `moreover, though emerging economies are also experiencing acute growth slowdowns, their share of the global economic pie will continue to grow.`
- len=11 span=(1,8) exact=True
  - observed: `in is china what precisely is this, fact experiencing.`
  - restored: `in fact, this is precisely what china is experiencing.`
  - target: `in fact, this is precisely what china is experiencing.`
- len=18 span=(4,7) exact=True
  - observed: `china, in particular the in western more become would, way that it manages its economy.`
  - restored: `china, in particular, would become more western in the way that it manages its economy.`
  - target: `china, in particular, would become more western in the way that it manages its economy.`
- len=16 span=(13,3) exact=True
  - observed: `now, private property will no longer be inferior to state property at. officially least`
  - restored: `now, private property will no longer be inferior to state property at least officially.`
  - target: `now, private property will no longer be inferior to state property at least officially.`
- len=25 span=(5,5) exact=True
  - observed: `there is a further major fund a to advantage global like the aiib: right now, the world suffers from insufficient aggregate demand.`
  - restored: `there is a further major global advantage to a fund like the aiib: right now, the world suffers from insufficient aggregate demand.`
  - target: `there is a further major global advantage to a fund like the aiib: right now, the world suffers from insufficient aggregate demand.`
- len=24 span=(15,7) exact=True
  - observed: `replacing coal with natural gas does reduce greenhouse gas emissions, even if natural gas long the in sustainable not is itself term.`
  - restored: `replacing coal with natural gas does reduce greenhouse gas emissions, even if natural gas itself is not sustainable in the long term.`
  - target: `replacing coal with natural gas does reduce greenhouse gas emissions, even if natural gas itself is not sustainable in the long term.`
- len=17 span=(1,4) exact=True
  - observed: `but who britons young the voted overwhelmingly to remain a part of europe almost certainly will.`
  - restored: `but the young britons who voted overwhelmingly to remain a part of europe almost certainly will.`
  - target: `but the young britons who voted overwhelmingly to remain a part of europe almost certainly will.`
- len=25 span=(9,7) exact=True
  - observed: `we expect these programs accessible to young people in significant attract will beyond and countries arab interest from students and strong support from employers.`
  - restored: `we expect these programs accessible to young people in arab countries and beyond will attract significant interest from students and strong support from employers.`
  - target: `we expect these programs accessible to young people in arab countries and beyond will attract significant interest from students and strong support from employers.`

## Boundary

Perturbation is local TreeHeap `Flip(span_root, full_depth)`, not external array
reverse. The span is given to the model, so this does not prove automatic
node/depth discovery.
