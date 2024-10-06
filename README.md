# Spotify-Advanced-Data-Analysis
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```
```python
sns.set_style("darkgrid")
df = pd.read_csv("data.csv")
df.head()
```
<img width="739" alt="Screenshot 2024-10-07 at 0 25 51" src="https://github.com/user-attachments/assets/e9020dcb-f682-4712-81c2-fc32bab0fe01">

## Data Cleaning
```python
df = df.fillna(0)
df.isna().sum()
```
<img width="184" alt="Screenshot 2024-10-07 at 0 26 11" src="https://github.com/user-attachments/assets/93d7504b-5843-4e00-a14b-e22520ff815f">

```python
df.info()
```
<img width="379" alt="Screenshot 2024-10-07 at 0 26 30" src="https://github.com/user-attachments/assets/5c0e74ca-3809-4b62-8d09-fac0f63cfe3c">

```python
df.columns
```
<img width="599" alt="Screenshot 2024-10-07 at 0 26 45" src="https://github.com/user-attachments/assets/214b77d8-ef4c-4d2b-ad28-543a483838c4">

```python
len(df.columns)
```
17
```python
df.describe()
```
<img width="727" alt="Screenshot 2024-10-07 at 0 27 10" src="https://github.com/user-attachments/assets/9cbda156-c755-46a8-b61b-c3e826ef3168">

##  Data Analysis
### Top 5 most popular artists¶
```python
top_five_artists = df.groupby("artist").count().sort_values(by = "song_title", ascending = False)["song_title"][:5]
top_five_artists
```
<img width="253" alt="Screenshot 2024-10-07 at 0 28 14" src="https://github.com/user-attachments/assets/cb571365-40c5-4507-997f-aef369b04914">

```python
top_five_artists.plot.barh()
plt.show()
```
<img width="657" alt="Screenshot 2024-10-07 at 0 28 33" src="https://github.com/user-attachments/assets/1f7c7545-f049-4a4f-8eb7-08f0133b1dbc">

```python
top_5_loudest_tracks = df[["loudness","song_title"]].sort_values(by="loudness", ascending=True)[:5]
top_5_loudest_tracks
```
<img width="417" alt="Screenshot 2024-10-07 at 0 28 50" src="https://github.com/user-attachments/assets/36041fc9-5e8d-46d9-af5f-683e4a12575c">

### Artist with the most danceability songs¶
```python
top_five_dance_songs = df[["danceability", "song_title", "artist"]].sort_values(by="danceability", ascending = False)[:5]
top_five_dance_songs
```
<img width="420" alt="Screenshot 2024-10-07 at 0 29 16" src="https://github.com/user-attachments/assets/23175b51-5d3a-4ad6-bacc-f483c071c992">

```python
plt.figure(figsize = (12,7))
sns.barplot(x="danceability", y="artist", data = top_five_dance_songs)
plt.title("Artists with the most danceability songs")
plt.show()
```
<img width="742" alt="Screenshot 2024-10-07 at 0 29 35" src="https://github.com/user-attachments/assets/46ee7e30-98e8-4383-adac-dba9fed4088a">

### Multiple feature plots
```python
interest_feature_cols = ["tempo","loudness","acousticness","danceability","duration_ms","energy","instrumentalness","speechiness","valence"]
```

```python
for feature_col in interest_feature_cols:
    pos_data = df[df["target"] == 1][feature_col]
    neg_data = df[df["target"] == 0][feature_col]
    
    plt.figure(figsize=(12,7))


    sns.histplot(pos_data, bins=30, label="Positive", color="red")
    sns.histplot(neg_data, bins=30, label = "Negative", color = "blue")

    plt.legend(loc="upper right")
    plt.title(f"Positive and Negative Plot for {feature_col}")
    plt.show()
```
<img width="742" alt="Screenshot 2024-10-07 at 0 30 31" src="https://github.com/user-attachments/assets/98598661-2117-46f5-b5db-10c7938f4582">
<img width="735" alt="Screenshot 2024-10-07 at 0 30 47" src="https://github.com/user-attachments/assets/84f16bfa-0ecb-4bb6-9487-411e63d4d099">
<img width="734" alt="Screenshot 2024-10-07 at 0 30 58" src="https://github.com/user-attachments/assets/9b20d1ae-abbd-46c5-a7b9-e28f239aa686">
<img width="729" alt="Screenshot 2024-10-07 at 0 31 06" src="https://github.com/user-attachments/assets/02a6fa6a-72c2-4067-b439-8ea28a4adcd5">
<img width="742" alt="Screenshot 2024-10-07 at 0 31 15" src="https://github.com/user-attachments/assets/8674f8e6-b5cb-45e5-8e88-b89725f25868">
<img width="730" alt="Screenshot 2024-10-07 at 0 31 24" src="https://github.com/user-attachments/assets/f7c66874-6b35-472c-9d70-37de1fa25227">
<img width="736" alt="Screenshot 2024-10-07 at 0 31 32" src="https://github.com/user-attachments/assets/1f7f2737-39bf-4ec5-bdd6-263bda4e9e73">
<img width="744" alt="Screenshot 2024-10-07 at 0 31 38" src="https://github.com/user-attachments/assets/1092a671-0dbd-4262-96c0-1d4310be799a">
<img width="753" alt="Screenshot 2024-10-07 at 0 31 45" src="https://github.com/user-attachments/assets/40e295a2-58b1-4e09-9a1a-4ed5c908d9bd">





