# Task:
- Prior work focuses on querying videos for entities across multiple frames
- What we want to do: query videos for entities performing actions across multiple frames
 
# Milestones:
### 1. Download data and set up conda env (DONE)
- Create conda virtual environment for project
- Download ~5G/9hr `jackson-town-square-2017-12-14.mp4` video and corresponding CSV with bounding boxes to directory called `231n/`
- Install opencv-python
- Download [FFMPEG](https://www.ffmpeg.org/download.html) to `231n`
- Build from source with the following commands (long-ish process):
	
```
cd ffmpeg-4.0

./configure  --prefix=/usr/local --enable-gpl --enable-nonfree --enable-libass \
--enable-libfdk-aac --enable-libfreetype --enable-libmp3lame \
--enable-libtheora --enable-libvorbis --enable-libvpx --enable-libx264 --enable-libx265 --enable-libopus --enable-libxvid \
--samples=fate-suite/

make
```
 
### 2. Divide video into smaller ~5s clips (DONE)
- In a separate, empty directory called `231n/jackson-clips`, run: 
 
```
/path/to/231n/ffmpeg-4.0/ffmpeg -i /path/to/231n/jackson-town-square-2017-12-14.mp4 -vcodec copy -f segment -reset_timestamps 1 -map 0 %d.mp4
``` 

- Run `rm 0.mp4` and `rm 6491.mp4` in `jackson-clips`

### 3. Look at data (DONE)
- Run `git clone https://github.com/stanford-futuredata/swag-python.git` in `cs231n/`
- Run `pip install git+file:///path/to/231n/swag-python`
- Move `swag-python/swag` to `231n/231n-final-project/src`. **Alternatively, you can skip the three bullet points above and just `git pull`!**
- Read in video in clips with `swag.VideoCapture()` in `read-clips.py`

### 4. Label few frames (DONE)
- Label 2.5% of data (first 7500 frames) in `data/LabeledVideoData.csv`.
- Each frame has three labels: $vehicle$, $inDirection$, and $outDirection$.
- Possible values for $vehicle$: car (0), truck (1)
- Possible values for $inDirection$ and $outDirection$ (clockwise from bottom): front (0), right (1), back (2), left (3)

### 5. Look into CNN/RNN architectures (IN PROGRESS)
- Tubes can be derived from given frame bounding-box coordinates.
- Once we have these tubes (one per object, separated by unique object ID):
	- Consider only the spatial region that is within the tube
	- Apply region of interest/tube of interest pooling to ensure that tubes have the same cross-section (fixed-diameter tubes). **What about pooling across time too? To create fixed-time/fixed-length tubes?**
	- Now we have fixed-diameter tube, our task is to label the entire tube with a single action class (one of 32 possible classes from various permutations of $vehicle$, $inDirection$, and $outDirection$).
	- Baseline: fully-connected layers after tube.
	- **Apply an RNN to predict actions? Use a ResNet-based pretrained model?**
		- Use labels during training; predict on test.
	- Calculate tube-action assignment score based on actions alone, not spatial overlap within the tube.

### 6. Move on to weak supervision
