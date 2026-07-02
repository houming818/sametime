# S1 Sentence Flip Echo Probe

Claim: `S1-ECHO-SENT-C01`
Predict: `P-S1-ECHO-SENT01`
Host: `io.grepcode.cn`

## Result

pilot_pass: `True`

```text
hard_treeheap_closure_exact = 1.000000
learned_ood_exact = 0.964500
learned_ood_token_acc = 0.997818
learned_ood_edit_similarity = 0.997871
no_inverse_ood_exact = 0.000000
```

## Readable Examples

- len=10 exact=True
  - observed: `. nato transform would step bold a such, yes`
  - restored: `yes, such a bold step would transform nato.`
  - target: `yes, such a bold step would transform nato.`
- len=26 exact=True
  - observed: `. communists russian ethnic, sovereignty pro key and nationalists ukrainian communist non key between emerged convenience of alliance building state informal an, fact in`
  - restored: `in fact, an informal state building alliance of convenience emerged between key non communist ukrainian nationalists and key pro sovereignty, ethnic russian communists.`
  - target: `in fact, an informal state building alliance of convenience emerged between key non communist ukrainian nationalists and key pro sovereignty, ethnic russian communists.`
- len=22 exact=True
  - observed: `. israel except country any to aid military american of that exceeds firms us with contracts arms s india of value the`
  - restored: `the value of india s arms contracts with us firms exceeds that of american military aid to any country except israel.`
  - target: `the value of india s arms contracts with us firms exceeds that of american military aid to any country except israel.`
- len=32 exact=True
  - observed: `. substantially resistance antibiotic of rise the slowing, 80 nearly by antibiotics of use the reduce would agencies regulatory governmental by enacted be could which of both alone measures two these`
  - restored: `these two measures alone both of which could be enacted by governmental regulatory agencies would reduce the use of antibiotics by nearly 80, slowing the rise of antibiotic resistance substantially.`
  - target: `these two measures alone both of which could be enacted by governmental regulatory agencies would reduce the use of antibiotics by nearly 80, slowing the rise of antibiotic resistance substantially.`
- len=29 exact=True
  - observed: `. challenges of set unique a faces countries brics the of each governance poor and institutions weak, example for confront economies developing all almost that problems the beyond`
  - restored: `beyond the problems that almost all developing economies confront for example, weak institutions and poor governance each of the brics countries faces a unique set of challenges.`
  - target: `beyond the problems that almost all developing economies confront for example, weak institutions and poor governance each of the brics countries faces a unique set of challenges.`
- len=12 exact=True
  - observed: `. ruling british the of lesson second the to us brings this`
  - restored: `this brings us to the second lesson of the british ruling.`
  - target: `this brings us to the second lesson of the british ruling.`
- len=25 exact=True
  - observed: `. dream only can people many which of education of level a reaching child a been has result the, experience s family each whatever`
  - restored: `whatever each family s experience, the result has been a child reaching a level of education of which many people can only dream.`
  - target: `whatever each family s experience, the result has been a child reaching a level of education of which many people can only dream.`
- len=16 exact=False
  - observed: `. quickly adopted be can office s prosecutor the reforming legislation condition eu remaining only the`
  - restored: `the only remaining eu condition legislation reforming the systems s office can be adopted quickly.`
  - target: `the only remaining eu condition legislation reforming the prosecutor s office can be adopted quickly.`
- len=14 exact=True
  - observed: `. everyone for out run has time until, paralysis policy is result the`
  - restored: `the result is policy paralysis, until time has run out for everyone.`
  - target: `the result is policy paralysis, until time has run out for everyone.`
- len=32 exact=True
  - observed: `. change institutional of phase next the navigate they as leaders s country the to invaluable prove could framework theoretical his, development institutional s china on explicitly focused never north though`
  - restored: `though north never focused explicitly on china s institutional development, his theoretical framework could prove invaluable to the country s leaders as they navigate the next phase of institutional change.`
  - target: `though north never focused explicitly on china s institutional development, his theoretical framework could prove invaluable to the country s leaders as they navigate the next phase of institutional change.`
- len=8 exact=True
  - observed: `. crucial be will cooperation global, finally`
  - restored: `finally, global cooperation will be crucial.`
  - target: `finally, global cooperation will be crucial.`
- len=8 exact=True
  - observed: `. relations eu us for peril holds this`
  - restored: `this holds peril for us eu relations.`
  - target: `this holds peril for us eu relations.`
- len=18 exact=True
  - observed: `. future the in invest to chance the it afford world zero g the in advantages s america`
  - restored: `america s advantages in the g zero world afford it the chance to invest in the future.`
  - target: `america s advantages in the g zero world afford it the chance to invest in the future.`
- len=11 exact=True
  - observed: `. straightforward relatively are up catch economic promote to programs coherent`
  - restored: `coherent programs to promote economic catch up are relatively straightforward.`
  - target: `coherent programs to promote economic catch up are relatively straightforward.`

## Boundary

Perturbation is TreeHeap `Flip(root, full_depth)`, not external array reverse.
This proves sentence-level same-algebra flip echo, not translation or semantic
understanding.
