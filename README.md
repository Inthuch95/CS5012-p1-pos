# CS5012-p1-pos
## Parameter 1 - corpus
1 = conll2000 with default tagset </br>
2 = conll2000 with universal tagset </br>
3 = treebank with default tagset </br>
4 = treebank with universal tagset </br>
5 = brown with universal tagset </br>

## Parameter 2 - smoothing
-l = laplace smoothing </br>
-g = good-turing smoothing </br>

## Running
The program can be found in src folder. It can be executed with the command: </br>
python main.py corpus smoothing </br></br>
To run the program with conll2000 corpus with default tagset and laplace smoothing, you should use the command: </br>
python main.py 1 -l
