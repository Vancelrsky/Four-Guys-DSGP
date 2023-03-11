# Four-Guys-DSGP

## Question

How to automatically switch notification mode on mobile phone by using sensory data collected from smartphone and smartwatch？

## Data source

http://extrasensory.ucsd.edu

Publicly available: everyone is invited to download the dataset for free and use it (conditioned on citing our original paper).

## Datasheet

### Motivation

1. ***For what purpose was the dataset created?***

    Recording users' daily behaviour through apps and wearable devices.
2. ***Was there a specific task in mind?***

    Recognizing Detailed Human Context In-the-Wild
3. ***Was there a specific gap that needed to be filled? Please provide a description.***

    No.
4. ***Who created the dataset (e.g., which team, research group) and on behalf of which entity (e.g., company, institution, organization)?***

    This dataset was collected in 2015-2016 by Yonatan Vaizman and Katherine Ellis.
5. ***Who funded the creation of the dataset? If there is an associated grant, please provide the name of the grantor and the grant name and number.***

    The supervision is professor Gert Lanckriet.
6. ***Any other comments?***

    No.

### Composition

1. ***What do the instances that comprise the dataset represent (e.g., documents, photos, people, countries)? Are there multiple types of instances (e.g., movies, users, and ratings; people and interactions between them; nodes and edges)? Please provide a description.*** 

   The instances are mostly students (both undergraduate and graduate) and research assistants from the UCSD campus. The regular natural behavior of the instances are collected through sensors in their smart phone and smart watch.

2. ***How many instances are there in total (of each type, if appropriate)?*** 

   60 instance 34 iPhone users, 26 Android users. 34 female, 26 male. 56 right handed, 2 left handed, 2 defined themselves as using both. Diverse ethnic backgrounds (each user defined their "ethnicity" how they liked), including Indian, Chinese, Mexican, Caucasian, Filipino, African American and more.

3. ***Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set? If the dataset is a sample, then what is the larger set? Is the sample representative of the larger set (e.g., geographic coverage)? If so, please describe how this representativeness was validated/verified. If it is not representative of the larger set, please describe why not (e.g., to cover a more diverse range of instances, because instances were withheld or unavailable).*** 

   it is a sample of instances. The instances in the dataset are 60 users of ExtraSensory mobile application, and no tests were run to determine representativeness.

4. ***What data does each instance consist of? “Raw” data (e.g., unprocessed text or images) or features? In either case, please provide a description.***

   Accelerometer (Tri-axial direction and magnitude of acceleration. 40Hz for ~20sec.), gyroscope (Rate of rotation around phone's 3 axes. 40Hz for ~20sec.), magnetometer (Tri-axial direction and magnitude of magnetic field. 40Hz for ~20sec.), watch accelerometer (Tri-axial acceleration from the watch. 25Hz for ~20sec.), watch compass (Watch heading (degrees). nC samples (whenever changes in 1deg).), location	Latitude (longitude, altitude, speed, accuracies. nL samples (whenever changed enough).	), location (quick) (Quick location-variability features (no absolute coordinates) calculated on the phone.), audio (22kHz for ~20sec. Then 13 MFCC features from half overlapping 96msec frames.), audio magnitude	(Max absolute value of recorded audio, before it was normalized.	), phone state (App status, battery state, WiFi availability, on the phone, time-of-day.), additional (Light, air pressure, humidity, temperature, proximity. If available sampled once in session.) Additional measurements were recorded from pseudo-sensors - processed versions that are given by the OS: Calibrated version of gyroscope (tries to remove drift affect). Unbiased version of magnetometer (tries to remove bias of the magnetic field created by the phone itself). Gravitation direction (the magnitude is always 1G). User-generated acceleration (raw acceleration minus gravitation acceleration). Estimated orientation of the phone. Rotation vector or attitude vector at every time-point. Also Original context labels, Mood labels, Absolute location coordinates.

5. ***Is there a label or target associated with each instance? If so, please provide a description.***

   Yes, cleaned labels are OR_indoors, LOC_home, SITTING, PHONE_ON_TABLE, LYING_DOWN, SLEEPING, AT_SCHOOL, COMPUTER_WORK, OR_standing, TALKING	, LOC_main_workplace, WITH_FRIENDS, PHONE_IN_POCKET, FIX_walking, SURFING_THE_INTERNET, EATING, PHONE_IN_HAND, WATCHING_TV, OR_outside, PHONE_IN_BAG, OR_exercise, DRIVE_-*I_M_THE_DRIVER, WITH_CO-WORKERS, IN_CLASS, IN_A_CAR, IN_A_MEETING, BICYCLING, COOKING, LAB_WORK, CLEANING, GROOMING, TOILET, DRIVE*-*I_M_A_PASSENGER, DRESSING, FIX_restaurant, BATHING*-*SHOWER, SHOPPING, ON_A_BUS, AT_A_PARTY, DRINKING__ALCOHOL*, WASHING_DISHES	, AT_THE_GYM, FIX_running, STROLLING, STAIRS_-*GOING_UP, STAIRS*-_GOING_DOWN, SINGING, LOC_beach, DOING_LAUNDRY, AT_A_BAR, ELEVATOR.

6. ***Is any information missing from individual instances? If so, please provide a description, explaining why this information is missing (e.g., because it was unavailable). This does not include intentionally removed information, but might include, e.g., redacted text.***

   Only few users reported mood labels, so some of the mood label data files are filled with missing labels. Value of 'nan' if the example has no location coordinate measurements, or if all the location updates taken in this example have poor horizontal-accuracy (more than 200meter). not all sensors were available for recording at all times, so for some examples (timestamps) the measurement file may be missing or may be a dummy file containing just 'nan'.

7. ***Are relationships between individual instances made explicit (e.g., users’ movie ratings, social network links)? If so, please describe how these relationships are made explicit.***

   No.

8. ***Are there recommended data splits (e.g., training, development/validation, testing)? If so, please provide a description of these splits, explaining the rationale behind them.*** 

   This has a pre-generated partition of the 60 users in the data to 5-folds and prepared text files with the list of users (UUIDs) of the train set and test set of each fold.

9. ***Are there any errors, sources of noise, or redundancies in the dataset? If so, please provide a description.***

   There are cases where the user wrongfully applied an irrelevant label; and more commonly, cases where a relevant label wasn't reported by the user (e.g. at home). For those reasons the dataset conducted the cleaning and provide the cleaned-version of the labels. Labels with prefix 'OR_', 'LOC_', or 'FIX_' are processed versions of original labels.

10. ***Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g., websites, tweets, other datasets)? If it links to or relies on external resources, a) are there guarantees that they will exist, and remain constant, over time; b) are there official archival versions of the complete dataset (i.e., including the external resources as they existed at the time the dataset was created); c) are there any restrictions (e.g., licenses, fees) associated with any of the external resources that might apply to a dataset consumer? Please provide descriptions of all external resources and any restrictions associated with them, as well as links or other access points, as appropriate.***

    It is self-contained.

11. ***Does the dataset contain data that might be considered confidential (e.g., data that is protected by legal privilege or by doctor– patient confidentiality, data that includes the content of individuals’ non-public communications)? If so, please provide a description.*** 

    No.

12. ***Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety? If so, please describe why. If the dataset does not relate to people, you may skip the remaining questions in this section.***

    No.

13. ***Does the dataset identify any subpopulations (e.g., by age, gender)? If so, please describe how these subpopulations are identified and provide a description of their respective distributions within the dataset.***

    It contains overview of the gender, age, height, weight and Body mass index of the instances, but no specific record for each individuals.

14. ***Is it possible to identify individuals (i.e., one or more natural persons), either directly or indirectly (i.e., in combination with other data) from the dataset? If so, please describe how.***

    No.

15. ***Does the dataset contain data that might be considered sensitive in any way (e.g., data that reveals race or ethnic origins, sexual orientations, religious beliefs, political opinions or union memberships, or locations; financial or health data; biometric or genetic data; forms of government identification, such as social security numbers; criminal history)? If so, please provide a description.***

    No.

16. ***Any other comments?***

### Collection Process

1. ***How was the data associated with each instance acquired? Was the data directly observable (e.g., raw text, movie ratings), reported by subjects (e.g., survey responses), or indirectly inferred/derived from other data (e.g., part-of-speech tags, model-based guesses for age or language)? If the data was reported by subjects or indirectly inferred/derived from other data, was the data validated/verified? If so, please describe how.*** 

   The developer deployed the app on the personal smartphones of 60 users and collected sensor data. The labels are reported by subjects.

2. ***What mechanisms or procedures were used to collect the data (e.g., hardware apparatuses or sensors, manual human curation, software programs, software APIs)? How were these mechanisms or procedures validated?***

   Data was collected using the ExtraSensory mobile application (see [the ExtraSensory App](http://extrasensory.ucsd.edu/ExtraSensoryApp)). We developed a version for iPhone and a version for Android, with a Pebble watch component that interfaces with both the iPhone and the Android versions. The app performs a 20-second "recording session" automatically every minute. In every recording session the app collects measurements from the phone's sensors and from the watch (if it is available). In addition, the app's interface is flexible and has many mechanisms to allow the user to report labels describing their activity and context

3. ***If the dataset is a sample from a larger set, what was the sampling strategy (e.g., deterministic, probabilistic with specific sampling probabilities)?*** 

   No, the dataset contain all possible instances

4. ***Who was involved in the data collection process (e.g., students, crowdworkers, contractors) and how were they compensated (e.g., how much were crowdworkers paid)?*** 

   The users were mostly students (both undergraduate and graduate) and research assistants from the UCSD campus.

5. ***Over what timeframe was the data collected? Does this timeframe match the creation timeframe of the data associated with the instances (e.g., recent crawl of old news articles)? If not, please describe the timeframe in which the data associated with the instances was created.*** 

   This dataset was collected in 2015-2016. The resulting Dataset is larger than previous datasets in scale (over 300,000 labelled minutes), range of behaviours (more than 50 diverse context-labels), and detail(combinations of more than three relevant labels per minute).

6. ***Were any ethical review processes conducted (e.g., by an institutional review board)? If so, please provide a description of these review processes, including the outcomes, as well as a link or other access point to any supporting documentation. If the dataset does not relate to people, you may skip the remaining questions in this section.*** 

   No, but there are no personal information contained in the dataset

7. ***Did you collect the data from the individuals in question directly, or obtain it via third parties or other sources (e.g., websites)?***

   From The ExtraSensory Dataset, [The ExtraSensory Dataset (ucsd.edu)](http://extrasensory.ucsd.edu/#papers)

8. ***Were the individuals in question notified about the data collection? If so, please describe (or show with screenshots or other information) how notice was provided, and provide a link or other access point to, or otherwise reproduce, the exact language of the notification itself.***

   No, but they clearly stated that the data is publicly available

9. ***Did the individuals in question consent to the collection and use of their data? If so, please describe (or show with screenshots or other information) how consent was requested and provided, and provide a link or other access point to, or otherwise reproduce, the exact language to which the individuals consented.***

   Yes, and conditioned on citing our original paper. [The ExtraSensory Dataset (ucsd.edu)](http://extrasensory.ucsd.edu/#papers)

10. ***If consent was obtained, were the consenting individuals provided with a mechanism to revoke their consent in the future or for certain uses? If so, please provide a description, as well as a link or other access point to the mechanism (if appropriate).***

    No, there is not a mechanism to revoke their consent in the future.

11. ***Has an analysis of the potential impact of the dataset and its use on data subjects (e.g., a data protection impact analysis) been conducted? If so, please provide a description of this analysis, including the outcomes, as well as a link or other access point to any supporting documentation.***

    There is not.

12. ***Any other comments?***

### Preprocessing/cleaning/labelling

1. ***Was any preprocessing/cleaning/labeling of the data done (e.g., discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)? If so, please provide a description. If not, you may skip the remaining questions in this section.***

   The data has been cleaned especially the labelled data, there are cases where the user wrongfully applied an irrelevant label; and more commonly, cases where a relevant label wasn't reported by the user (e.g. at home). For those reasons they conducted the cleaning and provide the cleaned-version of the labels.

2. ***Was the “raw” data saved in addition to the preprocessed/cleaned/labeled data (e.g., to support unanticipated future uses)? If so, please provide a link or other access point to the “raw” data.*** 

   - Accelerometer measurements: [Download the phone-accelerometer measurements zip file (6.1GB)](http://extrasensory.ucsd.edu/data/raw_measurements/ExtraSensory.raw_measurements.raw_acc.zip).
   - Gyroscope (calibrated) measurements: [Download the calibrated gyroscope measurements zip file (8.7GB)](http://extrasensory.ucsd.edu/data/raw_measurements/ExtraSensory.raw_measurements.proc_gyro.zip).
   - Magnetometer measurements: [Download the raw (uncalibrated) magnetometer measurements zip file (3.8GB)](http://extrasensory.ucsd.edu/data/raw_measurements/ExtraSensory.raw_measurements.raw_magnet.zip).
   - Watch-accelerometer measurements: [Download the watch-accelerometer measurements zip file (800MB)](http://extrasensory.ucsd.edu/data/raw_measurements/ExtraSensory.raw_measurements.watch_acc.zip).
   - Watch-compass measurements (heading): [Download the watch-compass measurements zip file (88MB)](http://extrasensory.ucsd.edu/data/raw_measurements/ExtraSensory.raw_measurements.watch_compass.zip).
   - Audio measurements: [Download the audio measurements zip file - this is not raw audio signals, but rather MFCCs computed on the phone (11GB)](http://extrasensory.ucsd.edu/data/raw_measurements/ExtraSensory.raw_measurements.audio.zip).
   - Gravity measurements: [Download the measurements of the estimated gravity component of the phone-acceleration. Zip file (9.4GB)](http://extrasensory.ucsd.edu/data/raw_measurements/ExtraSensory.raw_measurements.proc_gravity.zip).

3. ***Is the software that was used to preprocess/clean/label the data available? If so, please provide a link or other access point.***

   No

4. ***Any other comments?***

### Uses

1. ***Has the dataset been used for any tasks already? If so, please provide a description.*** 

   Yes, it has the original publication: 

   1. Vaizman, Y., Ellis, K., and Lanckriet, G. "Recognizing Detailed Human Context In-the-Wild from Smartphones and Smartwatches". IEEE Pervasive Computing, vol. 16, no. 4, October-December 2017, pp. 62-74. doi:10.1109/MPRV.2017.3971131

   2. Vaizman, Y., Weibel, N., and Lanckriet, G. "Context Recognition In-the-Wild: Unified Model for Multi-Modal Sensors and Multi-Label Classification". Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies (IMWUT), vol. 1, no. 4. December 2017. doi:10.1145/3161192

   3. Vaizman, Y., Ellis, K., Lanckriet, G., and Weibel, N. "ExtraSensory App: Data Collection In-the-Wild with Rich User Interface to Self-Report Behavior". Proceedings of the 2018 CHI Conference on Human Factors in Computing Systems (CHI 2018), ACM, April 2018. doi:10.1145/3173574.3174128

2. ***Is there a repository that links to any or all papers or systems that use the dataset? If so, please provide a link or other access point.***

   [Recognizing Detailed Human Context in the Wild from Smartphones and Smartwatches | IEEE Journals & Magazine | IEEE Xplore](https://ieeexplore.ieee.org/document/8090454)

   [Context Recognition In-the-Wild: Unified Model for Multi-Modal Sensors and Multi-Label Classification: Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies: Vol 1, No 4](https://dl.acm.org/doi/10.1145/3161192)

   [ExtraSensory App | Proceedings of the 2018 CHI Conference on Human Factors in Computing Systems (acm.org)](https://dl.acm.org/doi/10.1145/3173574.3174128)

3. ***What (other) tasks could the dataset be used for?*** 

   The ability to automatically recognize people’s behavioral context (the activities they’re doing, where they are, their body posture, etc.) is desirable for many domains, such as health management, aging care and office assistant systems

4. ***Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses? For example, is there anything that a dataset consumer might need to know to avoid uses that could result in unfair treatment of individuals or groups (e.g., stereotyping, quality of service issues) or other risks or harms (e.g., legal risks, financial harms)? If so, please provide a description. Is there anything a dataset consumer could do to mitigate these risks or harms?***

   There is nothing about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might affect future use.

5. ***Are there tasks for which the dataset should not be used? If so, please provide a description.***

   There are no tasks that should not use the dataset.

6. ***Any other comments?***

### Distribution

N.B this is probably quite a tricky set to answer but see if you can find information nonetheless.

1. ***Will the dataset be distributed to third parties outside of the entity (e.g., company, institution, organization) on behalf of which the dataset was created? If so, please provide a description.***

   This dataset is publicly available and everyone is welcome to download the dataset and use it for free.

2. ***How was the dataset distributed (e.g., tarball on website, API, GitHub)? Does the dataset have a digital object identifier (DOI)?***

   The dataset is distributed on this website: http://extrasensory.ucsd.edu/ and is available for download. The dataset does not have a DOI.

3. ***When was the dataset distributed?***

   The dataset was released in 2017.

4. ***Was the dataset distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)? If so, please describe this license and/or ToU, and provide a link or other access point to, or otherwise reproduce, any relevant licensing terms or ToU, as well as any fees associated with these restrictions.***

   The crawled data copyright belongs to the authors unless otherwise stated. There is no license and everyone is invited to download the dataset for free and use it, but there is a request to cite the corresponding paper if the dataset is used: Y. Vaizman, K. Ellis and G. Lanckriet, "Recognizing Detailed Human Context in the Wild from Smartphones and Smartwatches," in IEEE Pervasive Computing, vol. 16, no. 4, pp. 62-74, October-December 2017, doi: 10.1109/MPRV.2017.3971131.

5. ***Have any third parties imposed IP-based or other restrictions on the data associated with the instances? If so, please describe these restrictions, and provide a link or other access point to, or otherwise reproduce, any relevant licensing terms, as well as any fees associated with these restrictions.***

   No third parties impose IP-based or other restrictions on the data associated with the instances.

6. ***Do any export controls or other regulatory restrictions apply to the dataset or to individual instances? If so, please describe these restrictions, and provide a link or other access point to, or otherwise reproduce, any supporting documentation.***

   No. The dataset is publicly available only need to cite their original paper.

7. ***Any other comments?***

### Maintenance

NB again this may be difficult to answer, but should enable you to determine how the dataset has developed or if there are more recent versions of the data that you should be using.

1. ***Who will be supporting/hosting/maintaining the dataset?***

   This dataset was collected by Yonatan Vaizman and Katherine Ellis.

2. ***How can the owner/curator/manager of the dataset be contacted (e.g., email address)?***

   All comments can be sent toYonatan Vaizman: yvaizman@eng.ucsd.edu or yonatanv@gmail.com. 

3. ***Is there an erratum? If so, please provide a link or other access point.***

   Unknown.

4. ***Will the dataset be updated (e.g., to correct labeling errors, add new instances, delete instances)? If so, please describe how often, by whom, and how updates will be communicated to dataset consumers (e.g., mailing list, GitHub)?**

   Unknown.

5. ***If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances (e.g., were the individuals in question told that their data would be retained for a fixed period of time and then deleted)? If so, please describe these limits and explain how they will be enforced.***

   Unknown.

6. ***Will older versions of the dataset continue to be supported/hosted/maintained? If so, please describe how. If not, please describe how its obsolescence will be communicated to dataset consumers.***

   Unknown.

7. ***If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so? If so, please provide a description. Will these contributions be validated/verified? If so, please describe how. If not, why not? Is there a process for communicating/distributing these contributions to dataset consumers? If so, please provide a description.***

   Unknown.

8. ***Any other comments?***

## Plan
