# S1 Recursive TreeHeap Route Probe

Claim: `S1-RECURSIVE-ROUTE-C01`

This proof intentionally separates TreeHeap routing from a flat `L x L` route matrix.

A valid TreeHeap route here means:

```text
i = 1
K_theta(q, S_i, address_i) -> stop/left/right
i = 2*i or 2*i+1 until stop
```

It does not claim translation, semantics, or discovered natural-language syntax.

## Metrics

```json
{
  "train": {
    "sentences": 2000,
    "tokens": 21196,
    "oracle_exact": 1.0,
    "recursive_exact": 1.0,
    "flat_exact": 1.0,
    "recursive_token_acc": 1.0,
    "flat_token_acc": 1.0
  },
  "ood": {
    "sentences": 1842,
    "tokens": 51697,
    "oracle_exact": 1.0,
    "recursive_exact": 1.0,
    "flat_exact": 0.0,
    "recursive_token_acc": 1.0,
    "flat_token_acc": 0.009671741106834053
  },
  "recursive_train": {
    "trace": [
      {
        "epoch": 1,
        "loss": 0.8892619783679644,
        "step_acc": 0.7587076822916666
      },
      {
        "epoch": 2,
        "loss": 0.26767055767898756,
        "step_acc": 0.9378255208333334
      },
      {
        "epoch": 3,
        "loss": 0.018063500698190182,
        "step_acc": 1.0
      },
      {
        "epoch": 4,
        "loss": 0.0033000425901263952,
        "step_acc": 1.0
      },
      {
        "epoch": 5,
        "loss": 0.0018096726028791938,
        "step_acc": 1.0
      },
      {
        "epoch": 6,
        "loss": 0.0012753606473173325,
        "step_acc": 1.0
      },
      {
        "epoch": 7,
        "loss": 0.0009658632285815353,
        "step_acc": 1.0
      },
      {
        "epoch": 8,
        "loss": 0.0007585771648640124,
        "step_acc": 1.0
      },
      {
        "epoch": 9,
        "loss": 0.0006115343894634861,
        "step_acc": 1.0
      },
      {
        "epoch": 10,
        "loss": 0.0005046269931578232,
        "step_acc": 1.0
      },
      {
        "epoch": 11,
        "loss": 0.0004227424518224628,
        "step_acc": 1.0
      },
      {
        "epoch": 12,
        "loss": 0.0003595293528633192,
        "step_acc": 1.0
      },
      {
        "epoch": 13,
        "loss": 0.00030972823151387274,
        "step_acc": 1.0
      },
      {
        "epoch": 14,
        "loss": 0.0002697523238263481,
        "step_acc": 1.0
      },
      {
        "epoch": 15,
        "loss": 0.00023685053383815102,
        "step_acc": 1.0
      },
      {
        "epoch": 16,
        "loss": 0.00020963652180701806,
        "step_acc": 1.0
      },
      {
        "epoch": 17,
        "loss": 0.0001868969302449841,
        "step_acc": 1.0
      },
      {
        "epoch": 18,
        "loss": 0.00016768860587035306,
        "step_acc": 1.0
      },
      {
        "epoch": 19,
        "loss": 0.0001513940563503032,
        "step_acc": 1.0
      },
      {
        "epoch": 20,
        "loss": 0.00013723138908972032,
        "step_acc": 1.0
      },
      {
        "epoch": 21,
        "loss": 0.00012503212201409042,
        "step_acc": 1.0
      },
      {
        "epoch": 22,
        "loss": 0.00011438097953941906,
        "step_acc": 1.0
      },
      {
        "epoch": 23,
        "loss": 0.00010502461433740488,
        "step_acc": 1.0
      },
      {
        "epoch": 24,
        "loss": 9.67957530519925e-05,
        "step_acc": 1.0
      },
      {
        "epoch": 25,
        "loss": 8.944520808048158e-05,
        "step_acc": 1.0
      },
      {
        "epoch": 26,
        "loss": 8.294495910377009e-05,
        "step_acc": 1.0
      },
      {
        "epoch": 27,
        "loss": 7.709398657122317e-05,
        "step_acc": 1.0
      },
      {
        "epoch": 28,
        "loss": 7.18379060344887e-05,
        "step_acc": 1.0
      },
      {
        "epoch": 29,
        "loss": 6.714682604069822e-05,
        "step_acc": 1.0
      },
      {
        "epoch": 30,
        "loss": 6.284398114075884e-05,
        "step_acc": 1.0
      },
      {
        "epoch": 31,
        "loss": 5.896178860590832e-05,
        "step_acc": 1.0
      },
      {
        "epoch": 32,
        "loss": 5.5415105634892825e-05,
        "step_acc": 1.0
      },
      {
        "epoch": 33,
        "loss": 5.2167134072078625e-05,
        "step_acc": 1.0
      },
      {
        "epoch": 34,
        "loss": 4.9215112539968686e-05,
        "step_acc": 1.0
      },
      {
        "epoch": 35,
        "loss": 4.648441260239148e-05,
        "step_acc": 1.0
      },
      {
        "epoch": 36,
        "loss": 4.3994994863775595e-05,
        "step_acc": 1.0
      },
      {
        "epoch": 37,
        "loss": 4.1668933439117005e-05,
        "step_acc": 1.0
      },
      {
        "epoch": 38,
        "loss": 3.9524472110012233e-05,
        "step_acc": 1.0
      },
      {
        "epoch": 39,
        "loss": 3.7553721995209344e-05,
        "step_acc": 1.0
      },
      {
        "epoch": 40,
        "loss": 3.571608097748443e-05,
        "step_acc": 1.0
      },
      {
        "epoch": 41,
        "loss": 3.39874537379122e-05,
        "step_acc": 1.0
      },
      {
        "epoch": 42,
        "loss": 3.240886917410535e-05,
        "step_acc": 1.0
      },
      {
        "epoch": 43,
        "loss": 3.091687002173179e-05,
        "step_acc": 1.0
      },
      {
        "epoch": 44,
        "loss": 2.9538607956662116e-05,
        "step_acc": 1.0
      },
      {
        "epoch": 45,
        "loss": 2.823158623262619e-05,
        "step_acc": 1.0
      },
      {
        "epoch": 46,
        "loss": 2.7004630207253893e-05,
        "step_acc": 1.0
      },
      {
        "epoch": 47,
        "loss": 2.5871117410739924e-05,
        "step_acc": 1.0
      },
      {
        "epoch": 48,
        "loss": 2.4793343375980232e-05,
        "step_acc": 1.0
      },
      {
        "epoch": 49,
        "loss": 2.3781758803427994e-05,
        "step_acc": 1.0
      },
      {
        "epoch": 50,
        "loss": 2.282449501459875e-05,
        "step_acc": 1.0
      },
      {
        "epoch": 51,
        "loss": 2.192913401207382e-05,
        "step_acc": 1.0
      },
      {
        "epoch": 52,
        "loss": 2.1089029435946333e-05,
        "step_acc": 1.0
      },
      {
        "epoch": 53,
        "loss": 2.027427691094393e-05,
        "step_acc": 1.0
      },
      {
        "epoch": 54,
        "loss": 1.9522360768557217e-05,
        "step_acc": 1.0
      },
      {
        "epoch": 55,
        "loss": 1.8801022861225647e-05,
        "step_acc": 1.0
      },
      {
        "epoch": 56,
        "loss": 1.811912769274689e-05,
        "step_acc": 1.0
      },
      {
        "epoch": 57,
        "loss": 1.7466183635406196e-05,
        "step_acc": 1.0
      },
      {
        "epoch": 58,
        "loss": 1.684697144810343e-05,
        "step_acc": 1.0
      },
      {
        "epoch": 59,
        "loss": 1.6258865192260902e-05,
        "step_acc": 1.0
      },
      {
        "epoch": 60,
        "loss": 1.5706323362489154e-05,
        "step_acc": 1.0
      },
      {
        "epoch": 61,
        "loss": 1.5177485352069198e-05,
        "step_acc": 1.0
      },
      {
        "epoch": 62,
        "loss": 1.4665969236678697e-05,
        "step_acc": 1.0
      },
      {
        "epoch": 63,
        "loss": 1.4187256548818064e-05,
        "step_acc": 1.0
      },
      {
        "epoch": 64,
        "loss": 1.372895087570214e-05,
        "step_acc": 1.0
      },
      {
        "epoch": 65,
        "loss": 1.3290179064521604e-05,
        "step_acc": 1.0
      },
      {
        "epoch": 66,
        "loss": 1.287180536261682e-05,
        "step_acc": 1.0
      },
      {
        "epoch": 67,
        "loss": 1.2472092635107401e-05,
        "step_acc": 1.0
      },
      {
        "epoch": 68,
        "loss": 1.20850474255955e-05,
        "step_acc": 1.0
      },
      {
        "epoch": 69,
        "loss": 1.1719098438334186e-05,
        "step_acc": 1.0
      },
      {
        "epoch": 70,
        "loss": 1.1364836950633617e-05,
        "step_acc": 1.0
      },
      {
        "epoch": 71,
        "loss": 1.1029761859996748e-05,
        "step_acc": 1.0
      },
      {
        "epoch": 72,
        "loss": 1.0704861021319326e-05,
        "step_acc": 1.0
      },
      {
        "epoch": 73,
        "loss": 1.039884629487157e-05,
        "step_acc": 1.0
      },
      {
        "epoch": 74,
        "loss": 1.0095837031561436e-05,
        "step_acc": 1.0
      },
      {
        "epoch": 75,
        "loss": 9.81020086025334e-06,
        "step_acc": 1.0
      },
      {
        "epoch": 76,
        "loss": 9.538270281458002e-06,
        "step_acc": 1.0
      },
      {
        "epoch": 77,
        "loss": 9.270723921872559e-06,
        "step_acc": 1.0
      },
      {
        "epoch": 78,
        "loss": 9.015612325432206e-06,
        "step_acc": 1.0
      },
      {
        "epoch": 79,
        "loss": 8.768105487888533e-06,
        "step_acc": 1.0
      },
      {
        "epoch": 80,
        "loss": 8.529930009141632e-06,
        "step_acc": 1.0
      }
    ],
    "final_step_acc": 1.0,
    "final_loss": 8.529930009141632e-06
  },
  "flat_train": {
    "trace": [
      {
        "epoch": 1,
        "loss": 1.9954581148716763,
        "route_acc": 0.9720069289261735
      },
      {
        "epoch": 2,
        "loss": 0.3514618673669699,
        "route_acc": 1.0
      },
      {
        "epoch": 3,
        "loss": 0.10800405349239213,
        "route_acc": 1.0
      },
      {
        "epoch": 4,
        "loss": 0.06328393664157675,
        "route_acc": 1.0
      },
      {
        "epoch": 5,
        "loss": 0.04492329388521484,
        "route_acc": 1.0
      },
      {
        "epoch": 6,
        "loss": 0.03475324329093229,
        "route_acc": 1.0
      },
      {
        "epoch": 7,
        "loss": 0.028360663295894227,
        "route_acc": 1.0
      },
      {
        "epoch": 8,
        "loss": 0.02398778798669204,
        "route_acc": 1.0
      },
      {
        "epoch": 9,
        "loss": 0.020814780045558864,
        "route_acc": 1.0
      },
      {
        "epoch": 10,
        "loss": 0.01843975910162374,
        "route_acc": 1.0
      },
      {
        "epoch": 11,
        "loss": 0.016578846288512143,
        "route_acc": 1.0
      },
      {
        "epoch": 12,
        "loss": 0.015095016146214639,
        "route_acc": 1.0
      },
      {
        "epoch": 13,
        "loss": 0.013883260747297939,
        "route_acc": 1.0
      },
      {
        "epoch": 14,
        "loss": 0.012873280827073065,
        "route_acc": 1.0
      },
      {
        "epoch": 15,
        "loss": 0.012020258509360714,
        "route_acc": 1.0
      },
      {
        "epoch": 16,
        "loss": 0.011290644037785803,
        "route_acc": 1.0
      },
      {
        "epoch": 17,
        "loss": 0.010654854769768844,
        "route_acc": 1.0
      },
      {
        "epoch": 18,
        "loss": 0.010100513785634129,
        "route_acc": 1.0
      },
      {
        "epoch": 19,
        "loss": 0.00960932425720406,
        "route_acc": 1.0
      },
      {
        "epoch": 20,
        "loss": 0.009170288751455934,
        "route_acc": 1.0
      },
      {
        "epoch": 21,
        "loss": 0.008775562888830808,
        "route_acc": 1.0
      },
      {
        "epoch": 22,
        "loss": 0.008417675460231358,
        "route_acc": 1.0
      },
      {
        "epoch": 23,
        "loss": 0.00809329559309895,
        "route_acc": 1.0
      },
      {
        "epoch": 24,
        "loss": 0.00779529247859912,
        "route_acc": 1.0
      },
      {
        "epoch": 25,
        "loss": 0.00752045223367897,
        "route_acc": 1.0
      },
      {
        "epoch": 26,
        "loss": 0.007267313723159862,
        "route_acc": 1.0
      },
      {
        "epoch": 27,
        "loss": 0.00703072068643092,
        "route_acc": 1.0
      },
      {
        "epoch": 28,
        "loss": 0.0068108146940251855,
        "route_acc": 1.0
      },
      {
        "epoch": 29,
        "loss": 0.006604647061422056,
        "route_acc": 1.0
      },
      {
        "epoch": 30,
        "loss": 0.006411149470620241,
        "route_acc": 1.0
      },
      {
        "epoch": 31,
        "loss": 0.006228567973158523,
        "route_acc": 1.0
      },
      {
        "epoch": 32,
        "loss": 0.006057104677299213,
        "route_acc": 1.0
      },
      {
        "epoch": 33,
        "loss": 0.005893454222253559,
        "route_acc": 1.0
      },
      {
        "epoch": 34,
        "loss": 0.005738473651264045,
        "route_acc": 1.0
      },
      {
        "epoch": 35,
        "loss": 0.005591246074619175,
        "route_acc": 1.0
      },
      {
        "epoch": 36,
        "loss": 0.005450855595087928,
        "route_acc": 1.0
      },
      {
        "epoch": 37,
        "loss": 0.005316556207594025,
        "route_acc": 1.0
      },
      {
        "epoch": 38,
        "loss": 0.00518807526002712,
        "route_acc": 1.0
      },
      {
        "epoch": 39,
        "loss": 0.0050649794877909516,
        "route_acc": 1.0
      },
      {
        "epoch": 40,
        "loss": 0.004946817734061601,
        "route_acc": 1.0
      },
      {
        "epoch": 41,
        "loss": 0.004833414931908336,
        "route_acc": 1.0
      },
      {
        "epoch": 42,
        "loss": 0.0047239649708980475,
        "route_acc": 1.0
      },
      {
        "epoch": 43,
        "loss": 0.004618644643381497,
        "route_acc": 1.0
      },
      {
        "epoch": 44,
        "loss": 0.004517201080667676,
        "route_acc": 1.0
      },
      {
        "epoch": 45,
        "loss": 0.004419112279814076,
        "route_acc": 1.0
      },
      {
        "epoch": 46,
        "loss": 0.004324771966137839,
        "route_acc": 1.0
      },
      {
        "epoch": 47,
        "loss": 0.004232901181940436,
        "route_acc": 1.0
      },
      {
        "epoch": 48,
        "loss": 0.004144468364536121,
        "route_acc": 1.0
      },
      {
        "epoch": 49,
        "loss": 0.004058536438047719,
        "route_acc": 1.0
      },
      {
        "epoch": 50,
        "loss": 0.003975009808885096,
        "route_acc": 1.0
      },
      {
        "epoch": 51,
        "loss": 0.0038945063877401436,
        "route_acc": 1.0
      },
      {
        "epoch": 52,
        "loss": 0.003815949880438794,
        "route_acc": 1.0
      },
      {
        "epoch": 53,
        "loss": 0.0037397744279964304,
        "route_acc": 1.0
      },
      {
        "epoch": 54,
        "loss": 0.003665849398274203,
        "route_acc": 1.0
      },
      {
        "epoch": 55,
        "loss": 0.0035937222456646927,
        "route_acc": 1.0
      },
      {
        "epoch": 56,
        "loss": 0.0035236674410776465,
        "route_acc": 1.0
      },
      {
        "epoch": 57,
        "loss": 0.003455390678909159,
        "route_acc": 1.0
      },
      {
        "epoch": 58,
        "loss": 0.003389052891031666,
        "route_acc": 1.0
      },
      {
        "epoch": 59,
        "loss": 0.0033242905234071077,
        "route_acc": 1.0
      },
      {
        "epoch": 60,
        "loss": 0.0032613116878674343,
        "route_acc": 1.0
      },
      {
        "epoch": 61,
        "loss": 0.0031997313129424995,
        "route_acc": 1.0
      },
      {
        "epoch": 62,
        "loss": 0.0031398103380525077,
        "route_acc": 1.0
      },
      {
        "epoch": 63,
        "loss": 0.0030812401536467453,
        "route_acc": 1.0
      },
      {
        "epoch": 64,
        "loss": 0.003024123527505892,
        "route_acc": 1.0
      },
      {
        "epoch": 65,
        "loss": 0.002968293128784092,
        "route_acc": 1.0
      },
      {
        "epoch": 66,
        "loss": 0.0029137339614371824,
        "route_acc": 1.0
      },
      {
        "epoch": 67,
        "loss": 0.0028604712158526524,
        "route_acc": 1.0
      },
      {
        "epoch": 68,
        "loss": 0.0028084160447181044,
        "route_acc": 1.0
      },
      {
        "epoch": 69,
        "loss": 0.0027575892962720234,
        "route_acc": 1.0
      },
      {
        "epoch": 70,
        "loss": 0.0027077738478759167,
        "route_acc": 1.0
      },
      {
        "epoch": 71,
        "loss": 0.002659227256197319,
        "route_acc": 1.0
      },
      {
        "epoch": 72,
        "loss": 0.002611441390147127,
        "route_acc": 1.0
      },
      {
        "epoch": 73,
        "loss": 0.0025648522966552857,
        "route_acc": 1.0
      },
      {
        "epoch": 74,
        "loss": 0.002519144792594478,
        "route_acc": 1.0
      },
      {
        "epoch": 75,
        "loss": 0.0024746455775765657,
        "route_acc": 1.0
      },
      {
        "epoch": 76,
        "loss": 0.002430811700378549,
        "route_acc": 1.0
      },
      {
        "epoch": 77,
        "loss": 0.0023879623757339426,
        "route_acc": 1.0
      },
      {
        "epoch": 78,
        "loss": 0.0023460699683732706,
        "route_acc": 1.0
      },
      {
        "epoch": 79,
        "loss": 0.0023049375873266,
        "route_acc": 1.0
      },
      {
        "epoch": 80,
        "loss": 0.0022647369932339923,
        "route_acc": 1.0
      }
    ],
    "final_route_acc": 1.0,
    "final_loss": 0.0022647369932339923
  }
}
```

## Interpretation

- `oracle_exact` checks the hard TreeHeap mirror algebra.
- `recursive_exact` checks learned stop/left/right traversal.
- `flat_exact` is the old length-indexed route matrix baseline.
- This proof is only about the mechanism boundary: tree route vs matrix route.
