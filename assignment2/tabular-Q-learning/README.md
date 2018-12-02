The result isn't highly consistent as different runs often ends up with a average score below 0.78. Not sure if it's a hyperparameter problem. Currently, the `decay_rate` is applied to `epsilon` every 100 episodes. Not applying `decay_rate` doesn't seem to change the results much.

```
...
finished 4900 episodes
finished 5000 episodes
Q table (zeros correspond to terminating states (i.e. H, G):
[[0.17091602 0.14390981 0.15809765 0.14947282]
 [0.09350786 0.10609801 0.06123915 0.13935165]
 [0.12044589 0.11729601 0.1196288  0.12104846]
 [0.09772491 0.09297489 0.0897308  0.12089842]
 [0.20107275 0.15171536 0.09090047 0.12709168]
 [0.         0.         0.         0.        ]
 [0.09140576 0.10720156 0.13653838 0.05688309]
 [0.         0.         0.         0.        ]
 [0.15935186 0.17937224 0.22808683 0.29014499]
 [0.30256772 0.40813535 0.30473676 0.18990243]
 [0.43042526 0.29799769 0.21686086 0.17253485]
 [0.         0.         0.         0.        ]
 [0.         0.         0.         0.        ]
 [0.2816454  0.43730917 0.56648998 0.27447705]
 [0.49870026 0.84322869 0.72970288 0.63295136]
 [0.         0.         0.         0.        ]]
avg score over 1000: 0.816
```