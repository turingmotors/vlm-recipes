#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h=r10n4
#$ -l h_rt=24:00:00
#$ -o outputs/self_drain/$JOB_ID.log
#$ -e outputs/self_drain/$JOB_ID.log
#$ -p -5

./scripts/tsubame/infinite_sleep.sh
