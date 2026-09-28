# Gold-99 Claude Audit Report

clips audited: 99

## Consensus agreement with each source

| cue | vs manifest | vs FIG-mapped | vs parser (conf. subset) | A/B unanimity |
|---|---|---|---|---|
| flip | 83% (82/99) | 83% (82/99) | 88% (15/17) | 96% (95/99) |
| twist | 65% (64/99) | 65% (64/99) | n/a | 93% (92/99) |
| direction | 86% (85/99) | 84% (69/82) | 82% (14/17) | 93% (92/99) |
| context | 79% (78/99) | 79% (78/99) | n/a | 94% (93/99) |

## Per-clip verdicts

- ok: 71
- human_queue: 12
- disputed_ok: 11
- disputed_human_queue: 3
- claude_parser_agree_correction: 2

## Dispute/correction queue (for verify_server)

- `arabian` [human_queue] claude={'flip': 1, 'twist': 0.5, 'direction': 'backward', 'context': 'acrobatics'} manifest_flip=0.5 guess="arabian (half-twist back takeoff into front salto) off ledge to sand"
- `butterfly_twist` [human_queue] claude={'flip': 0, 'twist': 1, 'direction': 'none', 'context': 'acrobatics'} manifest_flip=1 guess="butterfly twist (b-twist)"
- `double_corkscrew` [human_queue] claude={'flip': 1, 'twist': 2, 'direction': 'backward', 'context': 'acrobatics'} manifest_flip=2 guess="double cork (running one-leg takeoff backward off-axis flip with 2 twists)"
- `double_frisbee` [human_queue] claude={'flip': 1, 'twist': 2, 'direction': 'forward', 'context': 'acrobatics'} manifest_flip=2 guess="double frisbee (flat 720 horizontal spin off edge)"
- `gainer_full` [disputed_ok] claude={'flip': 1, 'twist': 0, 'direction': 'backward', 'context': 'acrobatics'} manifest_flip=1 guess="gainer (tucked standing gainer off table edge into sand)"
- `gainer_triple_full` [disputed_ok] claude={'flip': 1, 'twist': 0, 'direction': 'backward', 'context': 'acrobatics'} manifest_flip=1 guess="gainer (flip off ledge into sand) — rotation count not observable"
- `gumbi` [disputed_human_queue] claude={'flip': 1, 'twist': 0.5, 'direction': 'side', 'context': 'acrobatics'} manifest_flip=0.5 guess="Gumbi (one-arm supported twisting side inversion)"
- `handstand_gainer` [human_queue] claude={'flip': 1.5, 'twist': 0, 'direction': 'backward', 'context': 'wall'} manifest_flip=1 guess="handstand gainer (off roof edge, to roll)"
- `lache_dark_arabian` [disputed_ok] claude={'flip': 1, 'twist': 0.5, 'direction': 'backward', 'context': 'swing'} manifest_flip=1 guess="Laché arabian (dark arabian): bar swing release, half twist into front salto"
- `macaco_in_back_out` [disputed_human_queue] claude={'flip': 2, 'twist': 0, 'direction': 'backward', 'context': 'wall'} manifest_flip=1 guess="macaco in back tuck out (macaco to back off drop)"
- `triple_corkscrew` [human_queue] claude={'flip': 1, 'twist': 3, 'direction': 'backward', 'context': 'acrobatics'} manifest_flip=4 guess="triple cork (cork with 3 twists)"
- `tunnel_flip` [disputed_ok] claude={'flip': 1, 'twist': 0, 'direction': 'forward', 'context': 'acrobatics'} manifest_flip=1 guess="standing tunnel flip off a drop (sideways-traveling front salto from roof edge)"
- `angel_drop` [human_queue] claude={'flip': 1, 'twist': 0, 'direction': 'forward', 'context': 'swing'} manifest_flip=0.5 guess="Angel drop (hanging drop with layout freefall into front tuck) from high bar"
- `gaet_pimp` [disputed_ok] claude={'flip': 1, 'twist': 0, 'direction': 'forward', 'context': 'wall'} manifest_flip=1 guess="Pimp flip (one-hand assisted front flip over wall/ledge)"
- `gaet_pimp_double_full` [disputed_human_queue] claude={'flip': 1, 'twist': 2, 'direction': 'backward', 'context': 'acrobatics'} manifest_flip=2 guess="gainer double full off a drop (pimp double full: back layout salto with 2 twists off elevated platform)"
- `palm_flip` [disputed_ok] claude={'flip': 1, 'twist': 0, 'direction': 'backward', 'context': 'wall'} manifest_flip=1 guess="palm flip (hand-assisted wall back flip)"
- `pimp_flip` [disputed_ok] claude={'flip': 1, 'twist': 0, 'direction': 'forward', 'context': 'acrobatics'} manifest_flip=1 guess="pimp flip (one-leg dive front flip off a drop)"
- `pop_castaway_back` [disputed_ok] claude={'flip': 1, 'twist': 0, 'direction': 'backward', 'context': 'acrobatics'} manifest_flip=1 guess="pop castaway backflip (cast on post, pop back tuck)"
- `wall_double_corkscrew` [disputed_ok] claude={'flip': 2, 'twist': 2, 'direction': 'backward', 'context': 'wall'} manifest_flip=2 guess="wall double cork (double corkscrew off wall)"
- `wall_gainer_full` [disputed_ok] claude={'flip': 1, 'twist': 1, 'direction': 'backward', 'context': 'wall'} manifest_flip=1 guess="wall gainer full (gainer full twist off tree/wall)"
- `wall_triple_corkscrew` [human_queue] claude={'flip': 1, 'twist': 3, 'direction': 'backward', 'context': 'wall'} manifest_flip=3 guess="wall triple cork (triple corkscrew off wall)"
- `flyaway_triple_full` [human_queue] claude={'flip': 1, 'twist': 3, 'direction': 'backward', 'context': 'swing'} manifest_flip=3 guess="Flyaway triple full (backward twisting layout dismount off bar)"
- `straddle_sole_circle_gainer` [human_queue] claude={'flip': 2, 'twist': 0, 'direction': 'backward', 'context': 'swing'} manifest_flip=1 guess="straddle sole circle to gainer (back tuck) dismount off bar"
- `swing_castaway_back_regrab` [human_queue] claude={'flip': 0.5, 'twist': 0, 'direction': 'backward', 'context': 'swing'} manifest_flip=1 guess="castaway back regrab (bar cast-away release and regrab)"
- `gate_vault` [human_queue] claude={'flip': 1, 'twist': 0, 'direction': 'forward', 'context': 'pk_basics'} manifest_flip=0 guess="gate vault — front flip over rail/fence with hand plant (dive kong / handspring-over)"
- `palm_spin` [disputed_ok] claude={'flip': 0, 'twist': 1, 'direction': 'none', 'context': 'acrobatics'} manifest_flip=0 guess="palm spin (palmspin / hand-plant pivot over a bollard)"