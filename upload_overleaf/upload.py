# overleaf_upload.py

import dropbox
from dropbox.files import WriteMode
from io import BytesIO
import subprocess
import matplotlib.pyplot as plt
import os
import requests


def upload(ax, project, path):
    """Upload plt or tex to Overleaf

    Parameters
    ----------
    ax : plt or tex
        The plot or table to upload to Overleaf. For plt, it should be a Matplotlib plot object.
    project : str
        The name of the Overleaf project to upload to
    path : str
        The path to the file to upload, including the file name and extension

    """
    bs = BytesIO()
    format = path.split(".")[-1]

    # Check if the file is a .tex file and handle it differently
    if format == "tex":
        # Do nothing
        pass
    else:
        ax.savefig(bs, bbox_inches="tight", format=format)

    # token = os.DROPBOX
    token = (
        subprocess.run(
            "curl https://api.dropbox.com/oauth2/token -d grant_type=refresh_token -d refresh_token=eztXuoP098wAAAAAAAAAAV4Ef4mnx_QpRaiqNX-9ijTuBKnX9LATsIZDPxLQu9Nh -u a415dzggdnkro3n:00ocfqin8hlcorr",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        .stdout.split('{"access_token": "')[1]
        .split('", "token_type":')[0]
    )
    dbx = dropbox.Dropbox(token)

    # Will throw an UploadError if it fails
    if format == "tex":
        # Handle .tex files by directly uploading their content
        dbx.files_upload(
            ax.encode(), f"/Apps/Overleaf/{project}/{path}", mode=WriteMode.overwrite
        )
    else:
        dbx.files_upload(
            bs.getvalue(), f"/Apps/Overleaf/{project}/{path}", mode=WriteMode.overwrite
        )
