# fault_tolerant_robot_design

![Image](https://github.com/user-attachments/assets/ef58947f-f6d2-4fcf-abd5-1ab7040ae07f)
A robot designed to resist free-swinging joint failures

![Image](https://github.com/user-attachments/assets/2d0ccdf1-b555-4a25-a89c-cc009144b33d)
A robot designed to resist locked joint failures

## Installation
Check [install.md](install.md).

## Demonstration
#### Options for `demo.py`
- `-t`, `--type`
  - `visualize`, `eval`, or `record`. Defaults to `visualize`.
- `-i`, `--initial_params_filename`
  - Path to the json file for the parameters.
- `-c`, `--cfg_filename`
  - cfg file name placed in `eagent/configs/*`.
- `-s`, `--search_cfg`
  - Whether to search for cfg files in the parent folder of `-i` or not.

`-i` should be set. Either `-c` or `-s` should be set.

#### Visualization
```bash
# Initial settigns of walking robot (for training by icm)
python demo.py -c ewalker.json -i zoo/walker/starfish6_48.64.json

# Trained walking robot: trained only by mdm
python demo.py -s 1 -i log/old/free_mdm_secondhalf_1/parameter_best.json

# Trained walking robot: trained only by icm
python demo.py -s 1 -i log/old/free_icm_secondhalf_1/parameter_best.json

# Trained walking robot: trained by mdm-icm (proposed method)
python demo.py -s 1 -i log/old/free_mdm_icm_secondhalf_1/parameter_best.json
```

#### Evaluation
```bash
python demo.py -t eval -s 1 -i log/old/free_mdm_secondhalf_1/parameter_best.json
```

The following command performs 5000 test runs with randomly selected failure patterns
```bash
python demo.py -t sim -s 1 -i log/old/free_mdm_secondhalf_1/parameter_best.json
```

#### Record video
```bash
# Probably requires ffmpeg
python demo.py -t record -s 1 -i log/old/free_mdm_secondhalf_1/parameter_best.json
```

## Perform training

#### Perform a new training in the walking task
The settings should be specified in a json file in `eagent/configs/ewalker_method`.
Some examples that we used are uploaded.
1. Edit the files as you like. In what follows, we assume that it is named as `my_cfg.json`.
   - Note that `eagent/configs/ewalker/default.json` is the default config, and `my_cfg.json` overrides `default.json`.
2. Execute the following command:
```bash
python train.py -c my_cfg.json
```
The training results are created in the `log` directory.

In the study, we employed 2-step learning. For example,
1. First, training with `ewalker_free_mdm_first_half.json` is performed.
2. Then, training with `ewalker_free_mdm_p_icm_second_half1.json` is performed using the log data generated by the first step.

#### Resume training
To resume training, set directory in the process with `-o` option. For example:
```bash
python train.py -c my_cfg.json -o `log/x.x.x_xxxxxxxx_xxxxxx`
```

It is necessary that a directory `log/x.x.x_xxxxxxxx_xxxxxx` exists and is trained using `my_cfg.json`. There must exist `log/x.x.x_xxxxxxxx_xxxxxx/checkpoint.json` that is automatically created and all the files described in it must exist. `.json` files are created for each `save_parameter_cycle`, and other files are created for each `checkpoint_cycle`.

#### Plot the training process
Use [plot_linked_history.ipynb](plot_linked_history.ipynb).
Note that the history data is large especially if icm is used. Therefore, we do not upload the data. If you want the data, please contact the corresponding author:
ryoariizumi@go.tuat.ac.jp

#### Important attributes in cfg
- `initial_params_filename`: It is necessary to edit `max_num_limbs`, `policy`, `policy_kwargs`, and `rl_cfg` to match `initial_params_filename`.
- `env_specific_cfg`:
  - `robot_cfg`:
    - `failure_occurrence`: This specifies whether actuator failures can occure or not.
    - `failure_type`: The type of possible actuator failure.
- `rl_cfg`: 
  - `algorithm`: `ppo`, `ddpg`, `ddpg_her`, or `sac_her`.
  - others: Passed to the model of stable_baselines3.
- `do_edges_selection`:
  - Whether to train **discrete morphological parameter**.
- `edges_selection_criteria`:
  - Valid only when `do_edges_selection: true`.
  - `fitting`, `contact_fitting`, or `fitting_rand`. `fitting` means **Isomorphic Classification Method**, `contact_fitting` means **Monotonic Decrease Method**.
- `edges_selection_params`:
  - Valid only when `do_edges_selection: true`.
  - The element with the name set in `edges_selection_criteria` is used.
- `do_structure_improvement`:
  - Whether to train **continuous morphological parameter**.
- `num_species`:
  - The number of workers.
- `num_individuals`:
  - The population in REINFORCE method.
- `max_generation`:
  - When the generation reaches this point, the training is terminated.

## License
```
Copyright (c) 2023 Kenta Kikuzumi
This software is released under the MIT License, see LICENSE.

Copyright (c) 2022 Ryosuke Koike
This software is released under the MIT License, see LICENSE.
```
