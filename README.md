# fault_tolerant_robot_design

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
# walking robot
python demo.py -c ewalker.json -i zoo/walker/starfish6_48.64.json

# manipulation robot
python demo.py -c ehand_egg.json -i zoo/hand/ehand5_below_sac_256.256.256.json

# Trained walking robot (these perform exactly the same)
python demo.py -s 1 -i log/old/0.8.8_20220203_225723/parameter_best.json
python demo.py -c ewalker_iso6.json -i log/old/0.8.8_20220203_225723/parameter_best.json

# Trained manipulation robot (these perform exactly the same)
python demo.py -s 1 -i log/old/0.8.8_20220205_213427/parameter_best.json
python demo.py -c ehand_egg_iso6.json -i log/old/0.8.8_20220205_213427/parameter_best.json
```

#### Evaluation
```bash
python demo.py -t eval -s 1 -i log/old/0.8.8_20220203_225723/parameter_best.json
```

#### Record video
```bash
# Probably requires ffmpeg
python demo.py -t record -s 1 -i log/old/0.8.8_20220203_225723/parameter_best.json
```

## Perform training
#### Reproduce in the prepared config
```bash
# Walking task
python train.py -c ewalker_iso6.json
python train.py -c ewalker_dec.json

# Manipulation task
python train.py -c ehand_egg_iso6.json
python train.py -c ehand_egg_dec.json
```

#### Perform a new training in the walking task
1. Clone `eagent/configs/ewalker/ewalker_iso6.json` to `eagent/configs/ewalker/my_cfg.json`.
2. Edit it as you like.
   - Note that `eagent/configs/ewalker/default.json` is the default config, and `my_cfg.json` overrides `default.json`.
3. Execute the following command:
```bash
python train.py -c my_cfg.json
```
The training results are created in the `log` directory.

#### Perform a new training in the manipulation task
Replace `ewalker` with `ehand_egg` in the previous section.

#### Resume training
To resume training, set directory in the process with `-o` option. For example:
```bash
python train.py -c my_cfg.json -o `log/x.x.x_xxxxxxxx_xxxxxx`
```

It is necessary that a directory `log/x.x.x_xxxxxxxx_xxxxxx` exists and is trained using `my_cfg.json`. There must exist `log/x.x.x_xxxxxxxx_xxxxxx/checkpoint.json` that is automatically created and all the files described in it must exist. `.json` files are created for each `save_parameter_cycle`, and other files are created for each `checkpoint_cycle`.

#### Plot the training process
Use [plot_history.ipynb](plot_history.ipynb).

#### Important attributes in cfg
- `initial_params_filename`: It is necessary to edit `max_num_limbs`, `policy`, `policy_kwargs`, and `rl_cfg` to match `initial_params_filename`.
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
