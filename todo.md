# Task:
- Prior work focuses on querying videos for entities across multiple frames
- What we want to do: query videos for entities performing actions across multiple frames

# To-Do's:
### 1. Download data and set up env
- Create conda virtual environment for project
- Download ~5G/9hr `jackson-town-square-2017-12-14.mp4` video and corresponding CSV with bounding boxes to directory called `cs231n/`
- Install opencv-python
- Download [FFMPEG](https://www.ffmpeg.org/download.html) to `cs231n`
- Build from source with the following commands (long-ish process):
	
```
cd ffmpeg-4.0

./configure  --prefix=/usr/local --enable-gpl --enable-nonfree --enable-libass \
--enable-libfdk-aac --enable-libfreetype --enable-libmp3lame \
--enable-libtheora --enable-libvorbis --enable-libvpx --enable-libx264 --enable-libx265 --enable-libopus --enable-libxvid \
--samples=fate-suite/

make
```
 
### 2. Divide video into smaller ~5s clips
- In a separate, empty directory called `cs231n/jackson-clips`, run: 
 
```
/path/to/cs231n/ffmpeg-4.0/ffmpeg -i /path/to/cs231n/jackson-town-square-2017-12-14.mp4 -vcodec copy -f segment -reset_timestamps 1 -map 0 %d.mp4
``` 

- Run `rm 0.mp4` and `rm 6491.mp4` in `jackson-clips`

### 3. Look at data
- Run `git clone https://github.com/stanford-futuredata/swag-python.git` in `cs231n/`
- Run `pip install git+file:///path/to/cs231n/swag-python`
- Move `swag-python/swag` to `cs231n/cs231n-final-project/src`. **Alternatively, you can skip the three bullet points above and just `git pull`!**
- Read in video in clips with `swag.VideoCapture()`

### 4. Label few frames

### 5. Look into CNN/RNN architectures
- T-CNNs? See literature/papers.md

### 6. Move on to weak supervision
