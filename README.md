<div align='center'>
<h2 align="center"> SpotLight: Robotic Scene Understanding through Interaction and Affordance Detection </h2>


<a href="">Tim Engelbracht</a><sup>1</sup>, <a href="https://scholar.google.com/citations?user=feJr7REAAAAJ&hl=en">René Zurbrügg</a><sup>1</sup>, <a href="https://people.inf.ethz.ch/marc.pollefeys/">Marc Pollefeys</a><sup>1,2</sup>, <a href="https://hermannblum.net/">Hermann Blum</a><sup>1,3</sup>, <a href="https://zuriabauer.com/">Zuria Bauer</a><sup>1</sup>

<sup>1</sup>ETH Zurich <sup>2</sup>Microsoft <sup>3</sup>Uni Bonn


![teaser](https://github.com/timengelbracht/SpotLight/blob/main/SpotLightLogo.png?raw=true)


</div>

[[Project Webpage](https://timengelbracht.github.io/SpotLight/)]



# Setup Instructions

The Spotlight code is based on the Spot-Compose codebase.
Please refer to the detailed setup instructions in the Spot-Compose repository [here](https://github.com/oliver-lemke/spot-compose). Since this repository is an "extension" to the Spot-Compose repository, the setup instructions are the same. To test the experiments presented in the paper, please run the following command:

```bash
python3 -m source/scripts/my_robot_scripts/light_switch_demo.py
```

For the application for updating scene graphs, please run the following:
```bash
python3 -m source/scripts/my_robot_scripts/scene_graph_demo.py
```

For swing door interaction refer to the following:
```bash
python3 -m source/scripts/my_robot_scripts/search_swing_drawer_dynamic.py
```

# Dataset

The dataset used in the paper is available at [Roboflow](https://universe.roboflow.com/timengelbracht/spotlight-light-switch-dataset)

# BibTeX :pray:
```
@misc{engelbracht2024spotlightroboticsceneunderstanding,
      title={SpotLight: Robotic Scene Understanding through Interaction and Affordance Detection}, 
      author={Tim Engelbracht and René Zurbrügg and Marc Pollefeys and Hermann Blum and Zuria Bauer},
      year={2024},
      eprint={2409.11870},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2409.11870}, 
}
```
