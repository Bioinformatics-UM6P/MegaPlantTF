if __name__ == "__main__":
    import sys
    from representation import DNA_MPI
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        domaine = sys.argv[2]
        k = int(sys.argv[3])
        output_file = sys.argv[4]
        DNA_MPI._build_kmer_representation_v2(input_file, domaine, k, output_file)
