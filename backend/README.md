# Stakeholder mapping MARL

The multi-agent RL problem is defined in a PettingZoo custom environment and trained with Tianshou v0.5.0.

## Files

Below is a list of files and their uses

### env/negotiation.py

This is PettingZoo custom environment, following the Agent-Environment-Cycle API.

### data/test.csv

This file contains the attributes of each stakeholder intended on being involved in a negotiation. The format is as follows:

Each row is its own agent, consisting of only the 5 attribute values
```
<position>,<power>,<knowledge>,<urgency>,<legitimacy>
```
The primary stakeholder is whichever stakeholder is in the first row, and the target stakeholder is whichever stakeholder is in the last row.

### test_negotiate.ipynb

The code to train on the RL environment. This is based on the Tianshou Tic-Tac-Toe tutorial, and probably contains more hyperparameter tunings than are necessary at the moment. It currently trains a policy on the RL environment, then steps through and prints outputs for one round of the negotiation.
