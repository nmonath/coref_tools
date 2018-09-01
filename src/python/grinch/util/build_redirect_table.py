import sys
import gzip

if __name__ == "__main__":
    redirect_file = "wiki-link/data/wiki/enwiki-20180820-redirect.tsv.gz"
    page_file = 'wiki-link/data/wiki/enwiki-20180820-page.tsv.gz'
    out_file = 'wiki-link/data/wiki/enwiki-20180820-redirect-table.tsv.gz'

    id2page = dict()
    with gzip.open(page_file,'rt') as fin:
        for line in fin:
            splt = line.strip().split("\t")
            pid = splt[0]
            name = splt[2]
            id2page[pid] = name

    redirects = dict()

    with gzip.open(redirect_file,'rt') as fin:
        for line in fin:
            splt = line.strip().split("\t")
            from_pid = splt[0]
            to_name = splt[2]
            if from_pid not in id2page:
                print('MISSING PID %s' % from_pid)
            else:
                from_name = id2page[from_pid]
                redirects[from_name] = to_name

    with gzip.open(out_file,'wt') as fout:
        for from_name,to_name in redirects.items():
            fout.write("%s\t%s\n" % (from_name, to_name))
            from_name_lower = from_name[0].lower() + from_name[1:]
            if from_name_lower != from_name:
                fout.write("%s\t%s\n" % (from_name_lower, to_name))
            to_name_lower = to_name[0].lower() + to_name[1:]
            if to_name_lower != to_name:
                fout.write("%s\t%s\n" % (to_name_lower, to_name))







