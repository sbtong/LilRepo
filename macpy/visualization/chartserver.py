#! /usr/bin/python

import sys
from optparse import OptionParser

import os.path
import os
import urllib
import cgi
import glob
try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO

import cherrypy

os_symlink = getattr(os, "symlink", None)
if callable(os_symlink):
    pass
else:
    def symlink_ms(source, link_name):
        import ctypes
        csl = ctypes.windll.kernel32.CreateSymbolicLinkW
        csl.argtypes = (ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32)
        csl.restype = ctypes.c_ubyte
        flags = 1 if os.path.isdir(source) else 0
        if csl(link_name, source, flags) == 0:
            raise ctypes.WinError()
    os.symlink = symlink_ms  #you can only run this with administrator privileges

class ChartServer(object):
    def __init__(self, basedir, static_name, extensions):
        self.basedir = basedir
        self.static_name = static_name
        self.extensions = extensions
        if self.extensions is None or len(self.extensions) == 0:
            self.extensions = ['html']

    @cherrypy.expose
    def index(self):
        return "Hello world!"

    def getPath(self, folders):
        if len(folders) > 0:
            pathname = os.path.join(self.basedir, *folders)
        else:
            pathname = self.basedir
        return pathname

    @cherrypy.expose
    def charts(self, *args):
        pathname = self.getPath(args)
        if os.path.isdir(pathname):
            return self.list_directory(args)
        else:
            return self.wrap_chart(args)

    def wrap_chart(self, folders):
        dirname = self.getPath(folders[:-1])
        thisfname = folders[-1]

        files = []
        for ext in self.extensions:
            files += [os.path.basename(x) for x in glob.glob(os.path.join(dirname, '*.' + ext))]
        files.sort(key=lambda a: a.lower())
        files = {fname: idx for idx, fname in enumerate(files)}
        revfiles = {idx: fname for fname, idx in files.iteritems()}
        curidx = files[thisfname]
        prevname = revfiles[(curidx - 1) % len(files)]
        nextname = revfiles[(curidx + 1) % len(files)]
        # print files
        # print thisfname, prevname, nextname
        wrapper_html = \
        """
        <html>
           <body>

              <iframe src="%s" height="90%%" width="100%%"></iframe>

              <form action="%s" style="float: left;">
                 <input type="submit" value="&larr;" />
              </form>
              <form action="%s">
                 <input type="submit" value="&rarr;" />
              </form>
              <form action="%s">
                 <input type="submit" value="Up" />
              </form>
           </body>
        </html>
        """
        base_url = cherrypy.request.base 
        wrapper_html = wrapper_html % ( (base_url + '/' + self.static_name + '/' + '/'.join(folders) ),
                (base_url + '/charts/' + '/'.join(list(folders[:-1]) + [prevname])),
                (base_url + '/charts/' + '/'.join(list(folders[:-1]) + [nextname])),
                (base_url + '/charts/' + '/'.join(folders[:-1])))
        # print wrapper_html
        return wrapper_html


    def list_directory(self, folders):
        """Helper to produce a directory listing (absent index.html).

        Return value is either a file object, or None (indicating an
        error).  In either case, the headers are sent, making the
        interface the same as for send_head().

        """
        path = self.getPath(folders)
        list = os.listdir(path)
        list.sort(key=lambda a: a.lower())
        f = StringIO()
        displaypath = cgi.escape(urllib.unquote(path))
        f.write('<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 3.2 Final//EN">')
        f.write("<html>\n<title>Directory listing for %s</title>\n" % displaypath)
        f.write("<body>\n<h2>Directory listing for %s</h2>\n" % displaypath)
        f.write("<hr>\n<ul>\n")
        for name in list:
            fullname = os.path.join(path, name)
            displayname = linkname = name
            # Append / for directories or @ for symbolic links
            if os.path.isdir(fullname):
                displayname = name + "/"
                linkname = cherrypy.request.base + '/charts/' + '/'.join(folders + (name,))
            else:
                if not any(name.endswith('.' + ext) for ext in self.extensions):
                    continue
                linkname = cherrypy.request.base + '/charts/' + '/'.join(folders + (name,))
            if os.path.islink(fullname):
                displayname = name + "@"
                # Note: a link to a directory displays with @ and links with /
            f.write('<li><a href="%s">%s</a>\n'
                    % (linkname, cgi.escape(displayname)))
        f.write("</ul>\n<hr>\n")
        previous = cherrypy.request.base + '/charts/' + '/'.join(folders[:-1])
        f.write('<a href="%s">Up</a>\n'
                % (previous,))
        f.write("</body>\n</html>\n")
        return f.getvalue()


def runmain(argv=None):
    if argv == None:
        argv = sys.argv

    usage = 'usage: %prog [options]\n'
    parser = OptionParser(usage=usage)
    parser.add_option("-p", "--port", dest="port", default=8080, type='int',
            help="Port to use (default: % default)", metavar="PORT")
    parser.add_option("--base_dir", dest="base_dir", 
            help="Base directory to use")
    parser.add_option("--ext", dest="extensions", action="append", default=['html'],
            help="Extensions to use, specify as many as needed (default: %default)")
    (cmdoptions, args) = parser.parse_args(argv)
    baseconfig = {'server.socket_port': cmdoptions.port,
              'server.socket_host': "0.0.0.0"}
    cherrypy.config.update(baseconfig)
    static_name = 'static'
    tmpfiles = glob.glob('static*')
    if len(tmpfiles) > 0:
        maxlen = max(len(x) for x in tmpfiles) - len(static_name) + 1
        static_name = static_name + '0' * maxlen
    os.symlink(cmdoptions.base_dir, static_name)
    config = { 
              '/': {"tools.staticdir.root": os.getcwd()},
              '/' + static_name: {
                "tools.staticdir.on": True,
                "tools.staticdir.dir": static_name 
                  }}

    cherrypy.quickstart(ChartServer(cmdoptions.base_dir, static_name, cmdoptions.extensions), '/', config)
    os.unlink(static_name)

if __name__ == "__main__":
    runmain()
