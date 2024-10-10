# Configurations for the LM experiments of the TMLR paper.


We provide configuration files to reproduce the experiments in the paper.
Each directory contains a configuration file `cfg.py` and a tensorboard
file `events.out.tfevents.v2` with the metrics of a reference run.

Please note that the DDS method is called 'soft' in the code.

| Directory   | Experiment Description |
| -------- | ------- |
| 4u9hqmpi3q  | c4lm_reuters mixed_training lr=0.00200 gen_w=0.900|
| af4ku9bvm3  | c4lm_reuters fine_tuning ft280_4u9hqmpi3q_mixing|
| i3tgyyyz7r  | c4lm_reuters soba downsampling=128 lr=0.00200|
| 2bmwrvrns9  | c4lm_reuters fine_tuning ft280_i3tgyyyz7r_soba|
| 5qcbc2tefm  | c4lm_reuters anograd downsampling=64 lr=0.00200|
| 9a2pe3643w  | c4lm_reuters fine_tuning ft280_5qcbc2tefm_anograd|
| a78b5uqumn  | c4lm_reuters soft downsampling=128 lr=0.00200|
| qi936um5hz  | c4lm_reuters fine_tuning ft280_a78b5uqumn_soft|
| vkxn8dzgt9  | c4lm_reuters cds lr=0.00200 cds pre=280k ft=100k|
| nrw6tzsarw  | c4lm_reuters fine_tuning ft280_vkxn8dzgt9_cds|
| xx9d5eudb8  | c4lm_reuters classifier downsampling=256.00|
| f6jfufih4w  | c4lm_reuters fine_tuning ft280_xx9d5eudb8_classifier|
| mq38hzu3wx  | c4lm_reuters standard_training lr=0.00200|
| phymwp4c6w  | c4lm_reuters fine_tuning lr=0.00200 ft280_mq38hzu3wx_base|
| 9tbccvzs5w  | c4lm_freelaw mixed_training lr=0.00200 gen_w=0.900|
| 9m5vmk8yxt  | c4lm_freelaw soba downsampling=128 lr=0.00200|
| qu6iwwexax  | c4lm_freelaw classif downsampling=256 lr=0.00200|
| t9ibs3zxr5  | c4lm_freelaw soft lr=0.00200 downsampling=128|
| ae4jwneuie  | c4lm_freelaw anograd lr=0.002 downsampling=32|
| y9ppueej65  | c4lm_freelaw cds num_pre=100000 downsampling=64 lr=0.00020|
| hhzj3ug3ay  | c4lm_freelaw standard_training lr=0.00200|
| 328n4jvr36  | c4lm_arxiv mixed_training lr=0.00200 gen_w=0.900|
| ckn4azgked  | c4lm_arxiv soba downsampling=128 lr=0.00200|
| ypysh48qxk  | c4lm_arxiv cds num_pre=160000 downsampling=64 lr=0.00200|
| x7wi98gv5r  | c4lm_arxiv anograd lr=0.002 downsampling=64|
| pgqsaypc6x  | c4lm_arxiv soft lr=0.00200 downsampling=128|
| he973ney7k  | c4lm_arxiv classifier downsampling=256|
| uxw5nuedap  | c4lm_arxiv standard_training lr=0.00200|
| k5wkcv294q  | c4lm_europarl mixed_training lr=0.00200 gen_w=0.950|
| 65aweieni5  | c4lm_europarl soba downsampling=128 lr=0.00200|
| qbjpmm78wy  | c4lm_europarl classif downsampling=256 lr=0.00200|
| qf97jkxeeb  | c4lm_europarl anograd lr=0.002 downsampling=128|
| 6stwju8iy2  | c4lm_europarl soft lr=0.00200 downsampling=32|
| 3b7jgzi5w5  | c4lm_europarl cds num_pre=100000 downsampling=32 lr=0.00200|
| f3c4ytshi2  | c4lm_europarl standard_training lr=0.00200|
| zk53jyz5j5  | c4lm_gutenberg_resplit mixed_training lr=0.00200 gen_w=0.900|
| i9ehwb84mk  | c4lm_gutenberg_resplit classif downsampling=256 lr=0.00200|
| riq25r8faj  | c4lm_gutenberg_resplit cds num_pre=160000 downsampling=64 lr=0.00200|
| ed46h2yzy8  | c4lm_gutenberg_resplit anograd lr=0.002 downsampling=64|
| ady2gtvip7  | c4lm_gutenberg_resplit soft lr=0.00200 downsampling=32|
| tjq2hch9h8  | c4lm_gutenberg_resplit soba downsampling=128 lr=0.00200|
| w4nhaque3j  | c4lm_gutenberg_resplit standard_training lr=0.00200|
| gct39i5g7z  | c4lm_opensubtitles mixed_training lr=0.00200 gen_w=0.900|
| bm7znyuxtt  | c4lm_opensubtitles soba downsampling=128 lr=0.00200|
| n5xpykzk2m  | c4lm_opensubtitles anograd lr=0.002 downsampling=64|
| k5mta6643w  | c4lm_opensubtitles classif downsampling=256 lr=0.00200|
| 3zyk2mtp73  | c4lm_opensubtitles anograd downsampling=32|
| 9nvmns2aiq  | c4lm_opensubtitles standard_training lr=0.00200|
| ygweqhme3s  | c4lm_opensubtitles soft downsampling=128|
| nmu47fqnxq  | c4lm_openwebtext2 classif downsampling=64 lr=0.00200|
| w9arpuux9s  | c4lm_openwebtext2 soba downsampling=32 lr=0.00200|
| e7r2mxxdyq  | c4lm_openwebtext2 anograd lr=0.002 downsampling=64|
| hk9mwf9tf4  | c4lm_openwebtext2 mixed_training lr=0.00200 gen_w=0.900|
| pjw4irtyyy  | c4lm_openwebtext2 cds num_pre=100000 downsampling=32 lr=0.00200|
| sigi8f4nkm  | c4lm_openwebtext2 standard_training lr=0.00200|
| 8pu3t3i7km  | c4lm_openwebtext2 soft downsampling=128|
| tyjiq6zk53  | c4lm_pubmed_abstracts classif downsampling=256 lr=0.00200|
| 5p4zmk227g  | c4lm_pubmed_abstracts soba downsampling=128 lr=0.00200|
| qd3gjvt8p8  | c4lm_pubmed_abstracts anograd lr=0.002 downsampling=128|
| kxgtdrk7ft  | c4lm_pubmed_abstracts soft lr=0.00200 downsampling=128|
| ztcknzxzid  | c4lm_pubmed_abstracts cds num_pre=100000 downsampling=32 lr=0.00200|
| fuzhujkwf3  | c4lm_pubmed_abstracts mixed_training lr=0.00200 gen_w=0.900|
| mbrguvas7w  | c4lm_pubmed_abstracts standard_training lr=0.00200|
| pvagfu7wt2  | c4lm_stackexchange mixed_training lr=0.00200 gen_w=0.900|
| ew4uc98emj  | c4lm_stackexchange cds num_pre=100000 downsampling=64 lr=0.00200|
| yynkke24dn  | c4lm_stackexchange soft lr=0.00200 downsampling=128|
| dedfr22i4e  | c4lm_stackexchange anograd lr=0.002 downsampling=128|
| itpjibdffv  | c4lm_stackexchange soba downsampling=128 lr=0.00200|
| gzy7ji759t  | c4lm_stackexchange classif downsampling=256 lr=0.00200|
| fqev62kdmi  | c4lm_stackexchange standard_training lr=0.00200|
| k5iyjkged3  | c4lm_wikipedia_en mixed_training lr=0.00200 gen_w=0.900|
| bejicyzx4j  | c4lm_wikipedia_en soba downsampling=128 lr=0.00200|
| 6qrmwhygrn  | c4lm_wikipedia_en classif downsampling=256 lr=0.00200|
| ve4ijzfc3m  | c4lm_wikipedia_en cds num_pre=100000 downsampling=32 lr=0.00200|
| r8hrvw5iq4  | c4lm_wikipedia_en soft lr=0.00200 downsampling=128|
| vq47ypmdvn  | c4lm_wikipedia_en anograd lr=0.002 downsampling=64|
| 6v9dhrdwn8  | c4lm_wikipedia_en standard_training lr=0.00200|
