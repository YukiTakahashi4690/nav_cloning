for i in `seq 11 11`
do
    echo "$i"
    rosrun nav_cloning offline_learning.py "$i"
    sleep 10s
done