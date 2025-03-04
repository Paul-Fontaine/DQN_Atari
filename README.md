# AI plays Atari games using Deep Reinforcement Learning

I trained an agent with a double DQN to play breakout

## packages installation
```pip install -r requirements.txt```  
```AutoRom --accept-license```
If you have a GPU, reinstall torch with the following command:  
```pip install torch --index-url https://download.pytorch.org/whl/cu126```  
Or visit [pytorch.org](https://pytorch.org/get-started/locally/) to install torch for your CUDA version.

## Usage

I already trained an agent and saved it in `agent_checkpoint.pth`
You can also train an agent with different hyperparameters by editing the `train.py` file and then run it ```python train.py```  

The `evaluate.py` file evaluate the agent with the average of the reawards it got on 10 episodes```python evaluate.py```  

Run `watch.py` file to watch the trained agent playing the game ```python watch.py```