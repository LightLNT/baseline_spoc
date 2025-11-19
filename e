[33mcommit a38849de58c908db59929d70b931e70534d68130[m[33m ([m[1;36mHEAD -> [m[1;32msparse_perception_h100[m[33m, [m[1;31morigin/sparse_perception_h100[m[33m)[m
Author: LightLNT <2517991693@qq.com>
Date:   Fri Nov 14 16:01:06 2025 +0800

    add rope to object_token_small and object_token_dinov2_small in obj extractor boundingbox

[33mcommit 5e679247ff29a8e998c2b81589d1d5425c278c71[m
Author: LightLNT <2517991693@qq.com>
Date:   Mon Nov 10 17:53:59 2025 +0800

    switch detect model to swib and input sensor modeify

[33mcommit 1c1170a09f12b9361d61d1b9df9eeb8e5658aedb[m
Author: LightLNT <2517991693@qq.com>
Date:   Fri Nov 7 09:51:14 2025 +0800

    some adaption to run gt training

[33mcommit f92ff29dad95d3ad59be41ba149e17168d1ed59e[m
Author: LightLNT <2517991693@qq.com>
Date:   Thu Nov 6 20:55:22 2025 +0800

    the prev commit delete the manipulate camera to reduce the inference cost and this push using gt to avoid inference in training

[33mcommit a9d9d57d1e89d5e5a561e362d1269376e0de03c0[m[33m ([m[1;32msparse_perception[m[33m)[m
Author: LightLNT <2517991693@qq.com>
Date:   Thu Nov 6 20:37:41 2025 +0800

    adaption in h100_using diff detect backbone

[33mcommit a7885afe4c35ab1bc9aa5fc8fb19cd2a6ce3918c[m[33m ([m[1;31morigin/sparse_perception[m[33m)[m
Author: LightLNT <2517991693@qq.com>
Date:   Thu Nov 6 06:15:15 2025 +0000

    modify_object_detection_backbone

[33mcommit d76c512fd8a4d4d083951dd6d0c3a31c9cc1bed3[m
Author: LightLNT <90267093+LightLNT@users.noreply.github.com>
Date:   Wed Nov 5 17:35:06 2025 +0800

    add_text_description_pass_to_detic and detic verify utils

[33mcommit 19c6afe6d0c7e783e398278bd83fedbcc682e2c6[m
Author: LightLNT <90267093+LightLNT@users.noreply.github.com>
Date:   Wed Nov 5 16:54:13 2025 +0800

    visualize_object_debug with detic result

[33mcommit 5346511c7227faafc6254462874bd904c60d79a5[m
Author: LightLNT <90267093+LightLNT@users.noreply.github.com>
Date:   Wed Nov 5 15:42:28 2025 +0800

    debug for visualize

[33mcommit ffe04972c860b0e418e2adfbaaaed6567f7114df[m
Author: LightLNT <2517991693@qq.com>
Date:   Wed Nov 5 01:31:30 2025 +0000

    sparse_perception with detic

[33mcommit 26f7d845e723e4739a146248dc629fa9ccd9e1b5[m
Author: LightLNT <2517991693@qq.com>
Date:   Tue Nov 4 13:07:38 2025 +0000

    connect groundingdino with task description

[33mcommit ac8597acb48827a189ffe396dccb6ef308a1f570[m
Author: LightLNT <2517991693@qq.com>
Date:   Tue Nov 4 11:33:17 2025 +0000

    sparse_perception with grounding_dino cause unbearable slow training

[33mcommit edcc66e689cdc8b0afd732a1c7e51263d7cb5fbe[m
Author: LightLNT <90267093+LightLNT@users.noreply.github.com>
Date:   Tue Nov 4 17:53:50 2025 +0800

    debug for slow training

[33mcommit bbcdf0135f7a9f08a8349d2f24fd7e3f21c9f314[m
Author: LightLNT <2517991693@qq.com>
Date:   Tue Nov 4 09:42:10 2025 +0000

    adaption for training

[33mcommit 839bb0f59bfd868a4807b420f51f8150c9ff5765[m
Author: LightLNT <90267093+LightLNT@users.noreply.github.com>
Date:   Tue Nov 4 15:17:38 2025 +0800

    sparse object token implement demo

[33mcommit 17589269b0317f2199446637c5bd5f04edf19f44[m
Author: LightLNT <2517991693@qq.com>
Date:   Sat Nov 1 03:48:42 2025 +0000

    train_eval_readme

[33mcommit baaa237ad162e31cafb0d920a18c31093a095065[m
Author: Rose Hendrix <51966472+rosehendrix@users.noreply.github.com>
Date:   Mon Oct 7 11:36:05 2024 -0700

    Update README.md

[33mcommit 1efeeaa8f862f7791712c42fddd7d3df7ced3d14[m
Author: Kiana Ehsani <ehsanik@gmail.com>
Date:   Thu Mar 28 16:43:19 2024 -0700

    Update README.md

[33mcommit e838eda5579e1f31bd3178af4713b88688ec71f2[m
Merge: 40989f8 ffcc5c0
Author: Kiana Ehsani <ehsanik@gmail.com>
Date:   Fri Mar 22 13:48:10 2024 -0500

    Merge pull request #18 from allenai/local_weight_loading
    
    Adding pretrained weigths, Fixing wandb logging, Reducing the local prints

[33mcommit ffcc5c065ef843d2e4252e3879f10aba706f9348[m
Merge: 912e740 3d4e79b
Author: Kiana Ehsani <ehsanik@gmail.com>
Date:   Fri Mar 22 13:47:46 2024 -0500

    Merge pull request #19 from allenai/example_bash_for_loca_online_eval
    
    add example bash

[33mcommit 3d4e79b52da61cb685fcbbe2de5070f1f2aee00d[m
Author: KuoHaoZeng <khzeng@allenai.org>
Date:   Fri Mar 22 11:43:29 2024 -0700

    add example bash

[33mcommit 40989f875bf1d36d7e69ea858879ef805def55b6[m
Merge: 09a321c 3f681ce
Author: Kiana Ehsani <ehsanik@gmail.com>
Date:   Fri Mar 22 12:43:02 2024 -0500

    Merge pull request #16 from allenai/fix_train_pl_ckpt_save_bug_from_main
    
    allow self.logger.experiment.id for online wandb

[33mcommit 912e7408bfc02363297b1f47f7bcb1624fe187ff[m
Author: Kiana Ehsani <ehsanik@gmail.com>
Date:   Fri Mar 22 10:40:29 2024 -0700

    fixing print and making table prints more clear

[33mcommit 3f681ce982b0f5799ca04ed1fb5f2c78576244cd[m
Author: KuoHaoZeng <khzeng@allenai.org>
Date:   Fri Mar 22 00:01:53 2024 -0700

    add ckpt downloader

[33mcommit 684109498b4659a8afd43bd494f612b7e22f128b[m
Author: KuoHaoZeng <khzeng@allenai.org>
Date:   Thu Mar 21 23:01:23 2024 -0700

    allow self.logger.experiment.id for online wandb

[33mcommit 09a321c8fc749a4006be10028405337272356b2e[m
Merge: 8602988 31b029c
Author: Kiana Ehsani <ehsanik@gmail.com>
Date:   Thu Mar 21 21:08:46 2024 -0500

    Merge pull request #14 from allenai/fix_augs
    
    fix augs bug

[33mcommit 31b029cb9dcd7bc41349c910f86c67bf0540e808[m
Author: KuoHaoZeng <khzeng@allenai.org>
Date:   Thu Mar 21 17:33:02 2024 -0700

    fix augs bug

[33mcommit 00e6552fb34bcf0dd66e3f1dee95e495b5c98c78[m
Author: Kiana Ehsani <ehsanik@gmail.com>
Date:   Thu Mar 21 16:34:06 2024 -0700

    fixing print

[33mcommit b20131ee8210110321207d1e947ff759a6e6dee0[m
Author: Kiana Ehsani <ehsanik@gmail.com>
Date:   Thu Mar 21 16:26:45 2024 -0700

    adding pretrained weigths, fixing wandb logging

[33mcommit 86029882922484799437ec566ca495cc489e641a[m
Merge: 008133f 8ba19c5
Author: Luca Weihs <astronomicalcuriosity@gmail.com>
Date:   Thu Mar 21 10:21:39 2024 -0700

    Merge pull request #11 from allenai/data_usage_notebook
    
    Added jupyter notebook for data use

[33mcommit 8ba19c5ba2fb832097849cc2617266aca7cac54b[m
Author: Jordi Salvador <jordis@allenai.org>
Date:   Thu Mar 21 18:20:39 2024 +0100

    removed ci workflow

[33mcommit 62db8831c0fbf1df50b7fc46539b32cb65ad3cab[m
Merge: 8393745 f6da611
Author: Jordi Salvador <jordis@allenai.org>
Date:   Thu Mar 21 18:15:22 2024 +0100

    Merge remote-tracking branch 'origin/jordis-ai2-patch-1' into data_usage_notebook

[33mcommit 83937451f5c17fde8aa478c4a9cef374581fcf7b[m
Merge: 05a3029 f07dcf8
Author: Jordi Salvador <jordis@allenai.org>
Date:   Thu Mar 21 18:15:16 2024 +0100

    Merge remote-tracking branch 'origin/resolve_setup' into data_usage_notebook

[33mcommit 05a30293fd198d03461afd45777fc8876233d2e6[m
Author: Jordi Salvador <jordis@allenai.org>
Date:   Thu Mar 21 10:38:34 2024 +0100

    Added jupyter notebook for data use
    
    This reverts commit 6c98edc288b3a9c7e00e73835b6255abaa8951b1.
    
    Added jupyter notebook for data use
    
    Delete ci.yml
    
    Minor edits
    
    Minor edit in README.md
    
    Fixed call to house downloading script

[33mcommit f6da611b3d280463f2fe09d895c3472e9cfcc7d6[m
Author: Jordi Salvador <39779720+jordis-ai2@users.noreply.github.com>
Date:   Thu Mar 21 18:08:34 2024 +0100

    Update setup.py

[33mcommit f07dcf807f42f81dc499dc518276aa8f0d3004ff[m
Author: Kiana Ehsani <ehsanik@gmail.com>
Date:   Thu Mar 21 10:01:25 2024 -0700

    formatting

[33mcommit 18a06481b2c5a23ed5a08b88994f9e37ffbdc2c4[m
Author: Kiana Ehsani <ehsanik@gmail.com>
Date:   Thu Mar 21 10:00:42 2024 -0700

    adding setup

[33mcommit 008133f64376ef8cc1707b58d2f04d700e0022f9[m
Author: Kiana Ehsani <ehsanik@gmail.com>
Date:   Tue Mar 19 16:35:06 2024 -0700

    hello world
    
    Co-authored-by: Kiana Ehsani <ehsanik@gmail.com>
    Co-authored-by: Rose Hendrix <roseh@allenai.org>
    Co-authored-by: Tanmay Gupta <tanmayg@allenai.org>
    Co-authored-by: KuoHaoZeng <khzeng@allenai.org>
    Co-authored-by: Luca Weihs <lucaw@allenai.org>
    Co-authored-by: Eli VanderBilt <40370237+elimvb@users.noreply.github.com>
    Co-authored-by: Jordi Salvador <jordis@allenai.org>
    Co-authored-by: Kunal Pratap Singh <krkunalking@gmail.com>
    Co-authored-by: AlvaroHG <alvaroh@allenai.org>
