for i in `seq 1 10`
do
    echo "$i"
    rosrun nav_cloning offline_learning_9cam.py "$i"
    # rosrun nav_cloning offline_learning_9cam_2.py "$i"
    sleep 10s
done