# 世界模型 · 蓝色结构框图 Blue block & structure diagrams  `1099-1104`

6 figures. Run any of them from `scripts/`; the PNG lands in `output/` under the same number.

| # | Script | What it draws | Use when |
|---|--------|---------------|----------|
| 1099 | `fig_wm_blue_hub_spoke.py` | `10.1109_access.2020.2975643__fig2_p4` — deep-blue hub, four mid-blue satellites, hairline grey connectors, white bold type, one warm accent link | Opening "the four open problems in X" slide or survey figure; reads at contact-sheet size |
| 1100 | `fig_wm_blue_infoflow.py` | `10.1109_access.2020.2983149__fig2_p4` — two-row (a)/(b) information flow, thick `#1c86b8` arrows, stacked cylinders for stores, black frame around the modular stage, feedback bar over the top of the canvas | Contrasting a modular pipeline against an end-to-end one; the shared x anchors make the collapse readable |
| 1101 | `fig_wm_blue_layered_stack.py` | The "system stack" idiom: five bars on one blue ramp (dark at the base), per-layer prose plus component chips, an abstraction arrow on the left and a right-hand failure-mode column | Any layered architecture where the ordering *is* the message; the ramp replaces a legend |
| 1102 | `fig_wm_blue_swimlane.py` | Three-lane swimlane in the flat blue dialect: alternating pale-blue washes, eleven solid-blue rounded steps spread over a uniform six-column grid (4 / 3 / 4 per lane), right-angle hand-offs, dashed warm outer loop under the bottom lane | Showing *ownership* of each step (which subsystem, not just which order); the vertical segments carry the meaning |
| 1103 | `fig_wm_blue_control_loop.py` | Textbook control block diagram: grey blocks with hairline edges, two true-circle summing junctions with +/- signs, mathtext transfer functions, thick cyan signal path, disturbance injection and a sensor feedback leg | Any feedback architecture; the dashed warm box is the reusable trick for "this one block is learned" |
| 1104 | `fig_wm_blue_venn_capability.py` | Three-disc Venn on one blue ladder, hairline leaders pulling the pairwise-lens labels out to the margin, only the triple intersection lettered in place | Positioning a contribution as "the place where three capabilities meet"; also the reference for *adjacent* ladder steps in a Venn |
