# Proxychains Note

## Problem shooting

* change the dns server of proxychains

    1. method 1: using environment variables

        ```bash
        export DNS_SERVER=8.8.8.8
        proxychains firefox
        ```

    1. method 2: chaning the bash file that proxychains uses

        file: `/usr/lib/proxychains3/proxyresolv`

        ```bash
        #!/bin/sh
        # This script is called by proxychains to resolve DNS names

        # DNS server used to resolve names
        DNS_SERVER=${PROXYRESOLV_DNS:-4.2.2.2}


        if [ $# = 0 ] ; then
                echo "  usage:"
                echo "          proxyresolv <hostname> "
                exit
        fi


        export LD_PRELOAD=libproxychains.so.3
        dig $1 @$DNS_SERVER +tcp | awk '/A.+[0-9]+\.[0-9]+\.[0-9]/{print $5;}'
        ```

        Reference: <https://blog.carnal0wnage.com/2013/09/changing-proxychains-hardcoded-dns.html>