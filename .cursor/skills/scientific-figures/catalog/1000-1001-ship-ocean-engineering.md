# 船舶与海洋工程 · Ship & ocean engineering  `1000-1001`

2 figures. Run any of them from `scripts/`; the PNG lands in `output/` under the same number.

| # | Script | What it draws | Use when |
|---|--------|---------------|----------|
| 1000 | `fig_shp_propeller_openwater.py` | Open-water thrust and torque coefficients and the efficiency computed as J KT over two pi KQ from those same arrays at three pitch ratios with the design point at the argmax and a back-cavitation wedge, beside best efficiency and its advance coefficient swept over pitch ratio, and hull load lines whose interpolated crossings give the operating points | 螺旋桨敞水性征讲解、桨型与螺距比选型、桨-船匹配工况点分析, "敞水特性, 推力系数, 扭矩系数, 敞水效率, 进速系数, 螺距比, 空泡限制, 负荷线, 工况点" |
| 1001 | `fig_shp_maneuvering_trials.py` | Equal-aspect turning-circle tracks at two rudder angles with advance and transfer interpolated at 90 degrees of heading change, tactical diameter at 180 degrees and a steady diameter from an algebraic circle fit, beside a 10/10 zigzag whose two overshoot angles are measured after each reversal, and the speed loss and yaw rate that built the track | 操纵性试验报告、回转圈与Z形试验讲解、航速衰减与偏航率分析, "回转试验, 战术直径, 纵距, 横距, Z形试验, 超越角, 航速衰减, 偏航率, 野本模型" |
