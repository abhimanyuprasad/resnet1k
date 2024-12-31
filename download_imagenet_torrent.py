import libtorrent as lt
import time
import os

def download_imagenet_torrent(save_path):
    """Download ImageNet using academic torrents"""
    # Torrent magnet links for ImageNet
    train_magnet = "magnet:?xt=urn:btih:5d6d0df7ed81715a363d9e14887f922f1d973dc2"
    val_magnet = "magnet:?xt=urn:btih:5d6d0df7ed81715a363d9e14887f922f1d973dc2"
    
    ses = lt.session()
    ses.listen_on(6881, 6891)
    
    print("Adding train torrent...")
    train_handle = lt.add_magnet_uri(ses, train_magnet, {'save_path': save_path})
    train_handle.set_sequential_download(True)
    
    print("Adding validation torrent...")
    val_handle = lt.add_magnet_uri(ses, val_magnet, {'save_path': save_path})
    val_handle.set_sequential_download(True)
    
    print("Downloading... This may take several hours.")
    while (not train_handle.is_seed()) or (not val_handle.is_seed()):
        s = train_handle.status()
        v = val_handle.status()
        
        print('\rTraining: {:.2f}% complete (down: {:.1f} MB/s up: {:.1f} MB/s peers: {}) '
              'Validation: {:.2f}% complete (down: {:.1f} MB/s up: {:.1f} MB/s peers: {})'.format(
                  s.progress * 100, s.download_rate / 1000000, s.upload_rate / 1000000, 
                  s.num_peers, v.progress * 100, v.download_rate / 1000000, 
                  v.upload_rate / 1000000, v.num_peers), end='')
        
        time.sleep(1)
    
    print("\nDownload complete!")

if __name__ == "__main__":
    download_imagenet_torrent("./imagenet") 