#!/usr/bin/env python
import json 
import re
import sys 

# sleep_pattern = set()

# with open("sleep_disorder.txt") as f:
#     for line in f:
#         sleep_pattern.add(re.compile(line.strip().lower()))

# for line in sys.stdin: 
#     try:
#         js = json.loads(line)
#     except: 
#         continue 

#     if not isinstance(js, dict):
#         continue

#     if 'posts' in js and js['posts']:
#         posts = js["posts"]
#     else:
#         continue
        
#     for post in posts:
#         flag = False
#         for pt in sleep_pattern:
#             if pt.search(post["text"].lower()):
#                 flag = True 
#                 break
#         if flag:
#             print(json.dumps(post))
mental_health_subreddit = ["depression_help","overcoming","add","cripplingalcoholism","disorder","Health","HealthProject","leaves","MenGetRapedToo","rapecounseling","7CupsofTea","addiction","ADHD","Advice","affirmations","afterthesilence","Agoraphobia","AlAnon","alcoholicsanonymous","alcoholism","Anger","Antipsychiatry","Anxiety","Anxietyhelp","anxietysuccess","anxietysupporters","ARFID","AskDocs","aspergers","AspiePartners","AtheistTwelveSteppers","behavior","behaviortherapy","bingeeating","BipolarReddit","BipolarSOs","BodyAcceptance","BPD","bulimia","CompulsiveSkinPicking","dbtselfhelp","depression_help","depressionregimens","disability","distractit","domesticviolence","downsyndrome","DysmorphicDisorder","eating_disorders","EatingDisorderHope","EatingDisorders","emetophobia","EOOD","ForeverAlone","fuckeatingdisorders","GetMotivated","getting_over_it","GFD","HaveHope","HealthAnxiety","helpmecope","itgetsbetter","leaves","MadOver30","mentalhealth","mentalillness","mentalpod","mixednuts","MMFB","MSTsurvivors","needadvice","Needafriend","neurodiversity","NoFap","nosurf","OCD","OCPD","offmychest","OpiatesRecovery","PanicParty","Phobia","PsychiatricFreedom","Psychiatry","psychology","psychopathology","psychotherapy","psychotic_features","psychoticreddit","ptsd","quittingkratom","rape","rapecounseling","reasonstolive","rehabtherapy","sad","schizoaffective","schizophrenia","secondary_survivors","selfharm","SelfHarmCommunity","selfhelp","siblingsupport","slp","SMARTRecovery","socialanxiety","socialskills","socialwork","socialworkresources","specialed","StopDipping","stopdrinking","stopgaming","StopSelfHarm","stopsmoking","https:wwww.reddit.comrstopspeeding","SuicideWatch","survivorsofabuse","swami","Teetotal","TheMixedNuts","tOCD","Tourettes","traumatoolbox","Trichsters","TwoXADHD","uniqueminds","whatsbotheringyou"]
for line in sys.stdin:
    try:
        post = json.loads(line)
    except:
        continue
    if isinstance(post,dict) and "subreddit" in post and "selftext" in post:
            if not post["subreddit"] in mental_health_subreddit and len(post["selftext"])>=10:
                new_post = {}
                new_post["title"] = post["title"].strip()
                new_post["subreddit"] = post["subreddit"]
                new_post["text"] = post["selftext"].strip()
                new_post["created_utc"] = post["created_utc"]
                print(json.dumps(new_post))
                # print(post['subreddit'])
 
