## M3u8 Note

## cache

* m3u8 中的 aes 解密

    下面是一些代码片段，无法直接运行，但是展示了解密 aes 的过程，可作为参考

    ```python
    pos = m3u8_url.find('//')
    pos = m3u8_url.find('/', pos + 2)
    main_domain = m3u8_url[:pos]  # https://www.xxx.com

    if line.startswith('#EXT-X-KEY'):
        is_ts_file_encrypted = True
        aes_iv = ''
        line_parts = line.split(',')
        for part in line_parts:
            if part.startswith('URI='):
                aes_key_url = part[5:].rstrip('\n')
                aes_key_url = aes_key_url.strip('"')
                if aes_key_url.startswith('/'):
                    aes_key_url = main_domain + aes_key_url
            if part.startswith('IV='):
                aes_iv = part[4:]
                aes_iv = aes_iv.lstrip('0x')
                aes_iv = aes_iv.rstrip('\n')
                
    aes_key_hex_cmd = ["xxd", '-p', '-c', '16', aes_key_file]
    exe_status = subprocess.run(aes_key_hex_cmd, capture_output=True, text=True)
    if exe_status.returncode != 0:
        print('fail to get aes key hex')
        return -1
    aes_key_hex = exe_status.stdout
    aes_key_hex = aes_key_hex.rstrip('\n')

    decrypt_cmd = ['openssl', 'aes-128-cbc', '-d', '-iv', aes_iv, '-K', aes_key_hex,
                   '-in', ts_file, '-out', 'tmp.ts']
    ret = subprocess.call(decrypt_cmd)
    if ret != 0:
        print('fail to decrypt, ret: {}'.format(ret))
    ```

* m3u8 format reference:

    <https://datatracker.ietf.org/doc/html/rfc8216#section-4.3.2.4>

## note
