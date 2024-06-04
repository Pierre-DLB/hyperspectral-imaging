import requests


def downloadfile(name, url):
    name = name + ".mp4"
    r = requests.get(url, stream=True)
    print("****Connected****")
    f = open(name, "wb")
    print("Donloading.....")
    for chunk in r.iter_content(chunk_size=255):
        if chunk:  # filter out keep-alive new chunks
            f.write(chunk)
    print("Done")
    f.close()


url_1 = (
    "https://d1yei2z3i6k35z.cloudfront.net/503601/6071f05eaf341_PROTOCOLEUSA1MAJ2.mp4"
)

if __name__ == "__main__":
    downloadfile("PROTOCOLEUSA1MAJ2", url_1)
