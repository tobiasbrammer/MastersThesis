# overleaf_upload.py

import dropbox
from dropbox.files import WriteMode
from io import BytesIO
import subprocess
import matplotlib.pyplot as plt


def upload(fig, project, path):
    """Upload plt or tex to Overleaf

    Parameters
    ----------
    fig : plt or tex
        The plot or table to upload to Overleaf. For plt, it should be a Matplotlib plot object.
    project : str
        The name of the Overleaf project to upload to
    path : str
        The path to the file to upload, including the file name and extension

    """
    bs = BytesIO()
    format = path.split(".")[-1]

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
            fig.encode(), f"/Apps/Overleaf/{project}/{path}", mode=WriteMode.overwrite
        )
    else:
        # Save the Matplotlib figure to the BytesIO object in the correct format
        # fig.savefig(bs, format=format)
        # Upload the BytesIO content (the plot image) to Dropbox
        # Format bs as a .png file
        fig.savefig(bs, dpi=250, format="png")
        dbx.files_upload(
            bs.getvalue(), f"/Apps/Overleaf/{project}/{path}", mode=WriteMode.overwrite
        )
