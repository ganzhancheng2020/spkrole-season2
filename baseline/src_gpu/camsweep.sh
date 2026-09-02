cd /root/autodl-tmp/ft
PY=/root/autodl-tmp/envs/moss/bin/python
for cfg in 0.02:6 0.10:6 0.05:4; do
  m=${cfg%%:*}; b=${cfg##*:}; d=output/_camwa_${m}_${b}
  [ -d "$d" ] && continue
  MARGIN=$m MAX_BLOCKS=$b $PY word_arb.py output/cam_re output/_camre_fr \
     /root/autodl-tmp/dev_wav "$d" >/tmp/cw_$m_$b.log 2>&1
  echo "CAMSWEEP_DONE m=$m b=$b"
done
tar czf camsweep.tgz output/_camwa_0.02_6 output/_camwa_0.10_6 output/_camwa_0.05_4 2>/dev/null
echo CAMSWEEP_ALL_DONE
