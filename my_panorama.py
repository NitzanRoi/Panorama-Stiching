import shutil
import os
import time
import sol4


def get_video_as_images():
    """
    saves videos as image sequences
    """
    experiments = ['me1.mp4']
    try:
        if (os.path.isdir("dump")):
            shutil.rmtree('dump')
    except OSError:
        print ("Deletion of the directory failed")
        exit()
    os.system('mkdir dump')
    for experiment in experiments:
        exp_no_ext = experiment.split('.')[0]
        subdir_cmd = "dump/{0}".format(exp_no_ext)
        os.mkdir(subdir_cmd)
        os.system('ffmpeg -i videos/%s dump/%s/%s%%03d.jpg' % (experiment, exp_no_ext, exp_no_ext))
        run_all(exp_no_ext)


def run_all(movie_name):
    """
    runs the panorama algorithm
    """
    s = time.time()
    panorama_generator = sol4.PanoramicVideoGenerator('dump/%s/' % movie_name, movie_name, 2100)
    panorama_generator.align_images(translation_only=False)
    panorama_generator.generate_panoramic_images(9)
    print(' time for %s: %.1f' % (movie_name, time.time() - s))
    panorama_generator.save_panoramas_to_video()


if __name__ == '__main__':
    get_video_as_images()
