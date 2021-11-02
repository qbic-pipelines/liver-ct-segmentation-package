=============================
liver-ct-segmentation-package
=============================

.. image:: https://github.com/luiskuhn/liver-ct-segmentation-package/workflows/Build%20liver-ct-segmentation-package%20Package/badge.svg
        :target: https://github.com/luiskuhn/liver-ct-segmentation-package/workflows/Build%20liver-ct-segmentation-package%20Package/badge.svg
        :alt: Github Workflow Build liver-ct-segmentation-package Status

.. image:: https://github.com/luiskuhn/liver-ct-segmentation-package/workflows/Run%20liver-ct-segmentation-package%20Tox%20Test%20Suite/badge.svg
        :target: https://github.com/luiskuhn/liver-ct-segmentation-package/workflows/Run%20liver-ct-segmentation-package%20Tox%20Test%20Suite/badge.svg
        :alt: Github Workflow Tests Status

.. image:: https://img.shields.io/pypi/v/liver-ct-segmentation-package.svg
        :target: https://pypi.python.org/pypi/liver-ct-segmentation-package
        :alt: PyPI Status


.. image:: https://readthedocs.org/projects/liver-ct-segmentation-package/badge/?version=latest
        :target: https://liver-ct-segmentation-package.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://flat.badgen.net/dependabot/thepracticaldev/dev.to?icon=dependabot
        :target: https://flat.badgen.net/dependabot/thepracticaldev/dev.to?icon=dependabot
        :alt: Dependabot Enabled


Prediction package for reproducible U-Net models trained on the LiTS dataset. These models are trained using the mlf-core framework (https://github.com/mlf-core/liver-ct-segmentation/).

* Free software: MIT
* Documentation: https://liverctsegmentationpackage.readthedocs.io.


Features
--------

* Model download CLI: liver-ct-seg-model-dl
* Prediction CLI: liver-ct-seg-pred
* Uncertainty of prediction CLI: liver-ct-seg-uncert
* Input feature importance (Guided Grad-CAM) CLI: liver-ct-seg-feat-ggcam

Credits
-------

This package was created with mlf-core_ using cookiecutter_.

.. _mlf-core: https://mlf-core.com
.. _cookiecutter: https://github.com/audreyr/cookiecutter
