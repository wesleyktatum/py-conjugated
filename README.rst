py-conjugated
=============

Machine learning on semiconducting polymer datasets

Background
----------

This repository compares the results of different regression techniques
on two different datasets:

-  Organic Photovoltaic (OPV) devices with the structure
   Glass/ITO/ZnO/P3HT:PC\ :math:`{60}`\ BM/:math:`MoO_{3}`/:math:`Ag`.
   JV curves were taken to determine power conversion efficiency (PCE),
   open-circuit voltage (V\ :math:`_oc}`), short-circuit current
   (J\ :math:`_{sc}`), and fill factor (FF). Force-distance mapping AFM
   images were taken of the active layer after JV-curve extraction.

-  Organic Field-Effect Transistor (OFET) devices with top-contact,
   bottom-gate architecture. P3HT was spin-coated onto
   :math:`Si/SiO_{2}` wafers passivated with octadecyl trichlorosilane
   (OTS). :math:`Au` source and drain electrodes, and :math:`In` ground
   contact. In the linear regime, the transfer curve was used to
   determine the hole-mobility (:math:`\mu_{lin}`), on-off ratio
   (:math:`on / off`), threshold voltage (:math:`V_{th}`), and the
   reliability coefficient (:math:`r`). Force-distance mapping AFM
   images were taken of the active layer between the source and drain
   electrodes after transfer curve extraction.

The force-distance maps were segmented and labeled using the `m2py
python library <https://github.com/ponl/m2py>`__. These label maps were
used to extract morphological information and, along with the device
performance data, is shown to be predictive of device performance. This
repository first compares the predictive capabilities of traditional
regression methods:

-  LASSO
-  Random Forrest
-  Support Vector Machines
-  Naive Bayes

In order to incorporate the image-likee morphological data, rather than
simply summaries thereof, Neural Networks are trained on the images and
m2py labels, as well as the tabular device data. Different combinations
of data and training schedules are used to evaluate the importance of
morphological labels in supervised models’ predictions of device
performance.

