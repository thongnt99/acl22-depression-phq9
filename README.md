# acl22-depression-phq9

## 1. Weakly label symptoms
### Collect positive symptoms
```
#!/bin/bash
symp_dir=('data/anhedonia'  'data/concentration'  'data/eating'  'data/fatigue'  'data/mood'  'data/psychomotor'  'data/self-esteem'  'data/self-harm'  'data/sleep')
pattern_dir='resources/patterns/'
pps=('anhedonia_problem.txt' 'concetration_problem.txt'  'eating_problem.txt'   'fatigue_problem.txt'  'mood_problem.txt'  'psychomotor_problem.txt'  'self-esteem_problem.txt'  'self-harm_problem.txt'  'sleep_disorder.txt')
np='negation_templates.txt'
cc='cc_templates.txt'
pt='pronoun_templates.txt'

for i in ${!symp_dir[*]}
do
        python prepare_data/filter_symptoms.py --input ${symp_dir[$i]}/posts  --output ${symp_dir[$i]}/filtered_post --pp $pattern_dir${pps[$i]} --np $pattern_dir$np --cc $pattern_dir$cc --pt $pattern_dir$pt --use_sentiment > ${symp_dir[$i]}/log.txt &
done
```
### Generate negative symptoms
```
python prepare_data/generate_neg.py --input input_file --ouptut ouptut_file [NEGATIVE_FOLDER]
```
### Aggregate and split folds 
```
python prepare_data/aggregate_and_split_data.py --input [INPUT_FOLDER]
```
## 2. Train question/symptom models
```
python code/train_question_model.py --epoch 100 --es_patience 5 --batch_size 16 --model cnn --train data/anhedonia/train.jsonl --dev data/anhedonia/dev.jsonl --test data/anhedonia/test.jsonl --save models/anhedonia --bert base --hidden_dim 5 --sig al 
```
## 3. Symptom inference on posts 
```
python code/run_questionnaire.py  --bert base --model cnn --hidden_dim 5 --input [INPUT_PATH]  --log [LOG_PATH] --output [OUTPUT_PATH]
```
## 4. Train depression models 
```
python code/train_depression.py --train [TRAIN_PATH] --dev [DEV_PATH] --test [TEST_PATH] --epochs 400 --es_patience 5 --batch_size 16 --lr 0.0001 --save models/ --model cnn --fold 0 
```
