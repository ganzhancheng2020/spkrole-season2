cd /root/autodl-tmp/ft
PY=/root/autodl-tmp/envs/moss/bin/python
d=output/_camwa_0.05_6
n0=$(ls $d/*.seglst.json 2>/dev/null|wc -l)
if [ "$n0" -ge 106 ]; then echo "SKIP 已有 $n0"; else
  echo "START m=0.05 b=6 $(date +%H:%M:%S)"
  MARGIN=0.05 MAX_BLOCKS=6 $PY word_arb.py output/cam_re output/_camre_fr \
     /root/autodl-tmp/dev_wav "$d" > camwa056.log 2>&1
  echo "DONE n=$(ls $d/*.seglst.json 2>/dev/null|wc -l) $(date +%H:%M:%S)"
fi
echo CAMWA056_ALL_DONE
