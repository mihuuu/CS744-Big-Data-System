#!/bin/bash

docker exec -it nn hdfs dfs -mkdir /task1/
docker exec -it nn hdfs dfs -put /enwiki-pages-articles/* /task1/
