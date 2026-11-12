#!/bin/bash
# Section 3.2: scripted teacher exit rate from the real start (84/100 at skill 3, 29/30 at skill 1).
# CPU only. The imitation runs also re-evaluate the teacher on the 100 held-out seeds.
source "$(dirname "$0")/common.sh"
mkdir -p $OUT/teacher
$PYTHON test_e1m1_teacher.py 3 100 | tee $OUT/teacher/skill3.log
$PYTHON test_e1m1_teacher.py 1 30  | tee $OUT/teacher/skill1.log
