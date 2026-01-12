import pstats
p = pstats.Stats('reports/vae_profile.txt')
p.sort_stats('cumulative').print_stats(10)