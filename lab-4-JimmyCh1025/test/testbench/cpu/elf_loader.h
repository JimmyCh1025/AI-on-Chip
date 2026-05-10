#ifndef ELF_LOADER_H
#define ELF_LOADER_H

#include <elf.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

/* ELF loader for RISC-V 64-bit binaries. Parses PT_LOAD segments and
 * the symbol table (.symtab/.strtab). BSS is zero-initialised. */
class ELFLoader {
   public:
    struct LoadedSegment {
        uint64_t vaddr;
        uint64_t paddr;
        size_t file_size;
        size_t mem_size;
        std::vector<uint8_t> data;
    };

    struct LoadedELF {
        uint64_t entry_point;
        std::vector<LoadedSegment> segments;
        std::map<std::string, uint64_t>
            symbols; /* symbol name -> virtual address */
    };

    static LoadedELF loadFromFile(const char* filename) {
        FILE* fp = fopen(filename, "rb");
        if (!fp) {
            throw std::runtime_error(std::string("Cannot open ELF file: ") +
                                     filename);
        }

        printf("Loading: %s\n", filename);

        Elf64_Ehdr ehdr;
        if (fread(&ehdr, sizeof(ehdr), 1, fp) != 1) {
            fclose(fp);
            throw std::runtime_error("Failed to read ELF header");
        }

        if (ehdr.e_ident[EI_MAG0] != ELFMAG0 ||
            ehdr.e_ident[EI_MAG1] != ELFMAG1 ||
            ehdr.e_ident[EI_MAG2] != ELFMAG2 ||
            ehdr.e_ident[EI_MAG3] != ELFMAG3) {
            fclose(fp);
            throw std::runtime_error("Invalid ELF magic");
        }

        if (ehdr.e_ident[EI_CLASS] != ELFCLASS64) {
            fclose(fp);
            throw std::runtime_error("Only 64-bit ELF is supported");
        }

        printf("entry=0x%lx, phdrs=%u, endian=%s\n", ehdr.e_entry, ehdr.e_phnum,
               ehdr.e_ident[EI_DATA] == ELFDATA2LSB ? "LE" : "BE");

        LoadedELF result;
        result.entry_point = ehdr.e_entry;

        for (int i = 0; i < ehdr.e_phnum; i++) {
            fseek(fp, ehdr.e_phoff + i * ehdr.e_phentsize, SEEK_SET);

            Elf64_Phdr phdr;
            if (fread(&phdr, sizeof(phdr), 1, fp) != 1) {
                throw std::runtime_error("Failed to read program header");
            }

            if (phdr.p_type != PT_LOAD) continue;

            printf(
                "PT_LOAD[%d]: vaddr=0x%lx, paddr=0x%lx, filesz=%lu, "
                "memsz=%lu\n",
                i, phdr.p_vaddr, phdr.p_paddr, phdr.p_filesz, phdr.p_memsz);

            LoadedSegment seg;
            seg.vaddr = phdr.p_vaddr;
            seg.paddr = phdr.p_paddr;
            seg.file_size = phdr.p_filesz;
            seg.mem_size = phdr.p_memsz;
            seg.data.resize(phdr.p_memsz, 0);

            if (phdr.p_filesz > 0) {
                fseek(fp, phdr.p_offset, SEEK_SET);
                if (fread(seg.data.data(), phdr.p_filesz, 1, fp) != 1) {
                    throw std::runtime_error("Failed to read segment data");
                }
            }

            result.segments.push_back(seg);
        }

        /* ── Parse symbol table (.symtab + .strtab) ── */
        if (ehdr.e_shnum > 0 && ehdr.e_shoff != 0) {
            /* Read all section headers */
            std::vector<Elf64_Shdr> shdrs(ehdr.e_shnum);
            fseek(fp, ehdr.e_shoff, SEEK_SET);
            for (int i = 0; i < ehdr.e_shnum; i++) {
                if (fread(&shdrs[i], sizeof(Elf64_Shdr), 1, fp) != 1) break;
            }

            /* Find .symtab and its associated .strtab */
            for (int i = 0; i < ehdr.e_shnum; i++) {
                if (shdrs[i].sh_type != SHT_SYMTAB) continue;

                Elf64_Shdr& symhdr = shdrs[i];
                Elf64_Shdr& strhdr = shdrs[symhdr.sh_link];

                /* Read string table */
                std::vector<char> strtab(strhdr.sh_size);
                fseek(fp, strhdr.sh_offset, SEEK_SET);
                if (fread(strtab.data(), strhdr.sh_size, 1, fp) != 1) continue;

                /* Read symbol table */
                size_t nsyms = symhdr.sh_size / sizeof(Elf64_Sym);
                std::vector<Elf64_Sym> syms(nsyms);
                fseek(fp, symhdr.sh_offset, SEEK_SET);
                if (fread(syms.data(), symhdr.sh_size, 1, fp) != 1) continue;

                for (size_t s = 0; s < nsyms; s++) {
                    if (syms[s].st_name == 0) continue;
                    const char* name = strtab.data() + syms[s].st_name;
                    result.symbols[std::string(name)] = syms[s].st_value;
                }

                printf("Parsed %zu symbols\n", result.symbols.size());
                break; /* only one .symtab expected */
            }
        }

        fclose(fp);

        printf("Loaded %zu PT_LOAD segments\n", result.segments.size());

        return result;
    }
};

#endif  // ELF_LOADER_H
