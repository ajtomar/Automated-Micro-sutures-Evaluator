
Micro-suturing is the task of performing suturing (stitching) under a microscope in the medical field. It is an indispensable neurosurgical skill. Automated evaluation of micro-suturing would lead to efficient, fast, and unbiased training of a trainee neurosurgeon. The dataset is available at \http://aiimsnets.org/microsuturing.asp We propose to evaluate the effectiveness (final image) of micro-suturing using various parameters. In this task, I have  calculated the three parameters of the evaluation using classical computer vision algorithms. Namely,

(a) Number of sutures

(b) Inter-suture distance

(c) Angulation of the suture

Below command needs to be provided for the execution of the program:

python3 main.py \<part_id\> \<img_dir\> \<output_csv\>

There are two tasks:

– For the first part, we are required to iterate through all the images in the given directory and generate a CSV file specifying the various parameters that you have calculated. It should have the following columns: image name, number of sutures, mean inter suture spacing, variance of inter suture spacing, mean suture angle wrt x-axis, variance of suture angle wrt x-axis. main.py will be executed as follows:

python3 main.py 1 \<img_dir\> \<output_csv\>


– For the second part, we will be given pairs of images and we need to output which image is a better suture outcome with respect to inter-suture distance and angulation of sutures. main.py will be executed as follows:

python3 main.py 2 \<input_csv\> \<output_csv\>

The input CSV file has the following two columns: img1 path, img2 path, where as output CSV should have the following four columns: img1 path, img2 path, output distance, output angle, where output distance and output angle take the values either 1 or 2, depending upon which image is better with respect to that feature.
