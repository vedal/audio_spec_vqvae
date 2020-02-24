import errno
import logging
import os
import sys
import zipfile
from pathlib import Path


def init_logging(logpath):
    # start python logging module
    logging.basicConfig(level=logging.DEBUG,
                        filename=logpath,
                        filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    logging.info('')

    fmt = "{elapsed_time:<25} {epoch:<25} {loss:<25} {min_loss_epoch:<20}"
    print_and_log(
        fmt.format(elapsed_time='elapsed time(min)',
                   epoch='epoch',
                   loss='loss-test',
                   min_loss_epoch='min-loss-epoch'))
    return fmt


def get_date_time_hparams_as_str(hparams):
    # format
    # <date>--<time>--<hparams-dict>
    # example:
    # 2018-09-26--T08-58-36--batch_size=8,lr=0.001
    return get_timestamp_str() + '--' + hparams_dict_to_str(hparams)


def hparams_dict_to_str(d):
    return ','.join('{}={}'.format(*i) for i in sorted(d.items()))


def get_timestamp_str():
    from time import strftime
    return str(strftime("%Y-%m-%d--T%H-%M-%S"))


def get_git_branch_and_hash():
    # gets branch name and last commit hash from current machine
    import subprocess
    try:
        git_branch_name = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
        git_branch_name = git_branch_name.strip().decode("utf-8")
        git_revision_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'])
        git_revision_hash.strip().decode("utf-8")
    except FileNotFoundError:
        return 0, 0

    return git_branch_name, git_revision_hash


def mkdir_p(path):
    # make folder recursively (mk -p)
    # https://stackoverflow.com/a/600612/6241937

    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def save_experiment_src_as_zip(zipdir):
    # ensure that target folder exists (recursive)
    zipdir = Path(zipdir)
    zipdir.mkdir(parents=True, exist_ok=True)

    zip_filepath = zipdir / 'code.zip'

    def zipdir(path, ziph):
        # ziph is zipfile handle
        for root, dirs, files in os.walk(path):
            for file in files:
                ziph.write(os.path.join(root, file))

    with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:

        # zip the 'src' dir
        print('zipping the src folder')
        zipdir('src/', zipf)

        current_filename = sys.argv[0]

        # add the current script to the zip file
        zipf.write(current_filename)
        print(current_filename, 'saved to zip archive', zip_filepath)

    return


def get_current_filename():
    return sys.argv[0].split('.')[0]


def print_and_log(s):
    logging.info(s)
    print(s)
    return
