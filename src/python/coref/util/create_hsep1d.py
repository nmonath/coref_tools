"""
Copyright (C) 2018 University of Massachusetts Amherst.
This file is part of "coref_tools"
http://github.com/nmonath/coref_tools
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
"""Invoke using sh bin/util/create_hsep1d.sh $LEVELS $OUT."""
import argparse
import errno
import json
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create hierarchical separated data.')
    parser.add_argument('levels', type=int,
                        help='Create 2^levels number of points.')
    parser.add_argument('--out', type=str, help='path to output file.')
    args = parser.parse_args()

    levels = args.levels
    out = args.out

    pts = []
    pt_seps = []
    for i in range(1, levels):
        pt_seps.append(i)
        pt_seps.extend(pt_seps[:-1])

    max_sep = max(pt_seps)
    curr = 0
    gt = 1
    pts.append((curr, gt))
    for sep in pt_seps:
        curr += sep
        if sep == max_sep:
            gt = 2
        pts.append((curr, gt))

    ms = []
    for pt, gt in pts:
        m = {'id': pt,
             'pack': ['d:%d' % pt],
             'attrs': ['d:%d' % pt],
             'gt': gt}
        ms.append(json.dumps(m))

    if out is not None:
        outf = os.path.join(out, 'ments.json')

        if not os.path.exists(os.path.dirname(outf)):
            try:
                os.makedirs(os.path.dirname(outf))
                with open(outf, 'w') as f:
                    for m in ms:
                        f.write('%s\n' % m)

            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
    else:
        for m in ms:
            print(m)

    print('[done.]')
