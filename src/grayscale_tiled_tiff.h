#pragma once
#include "abs_tile_loader.h"

#ifdef __APPLE__
#define uint64 uint64_hack_
#define int64 int64_hack_
#include <tiffio.h>
#undef uint64
#undef int64
#else
#include <tiffio.h>
#endif
#include <cstring>
#include <sstream>

constexpr size_t STRIP_TILE_HEIGHT = 1024;
constexpr size_t STRIP_TILE_WIDTH = 1024;
constexpr size_t STRIP_TILE_DEPTH = 1;

/// @brief Tile Loader for 2D Grayscale tiff files
/// @tparam DataType AbstractView's internal type
template<class DataType>
class NyxusGrayscaleTiffTileLoader : public AbstractTileLoader<DataType> 
{
public:

    /// @brief NyxusGrayscaleTiffTileLoader unique constructor
    /// @param numberThreads Number of threads associated
    /// @param filePath Path of tiff file
    NyxusGrayscaleTiffTileLoader(size_t numberThreads, std::string const& filePath)
        : AbstractTileLoader<DataType>("NyxusGrayscaleTiffTileLoader", numberThreads, filePath) 
    {
        short samplesPerPixel = 0;

        // Open the file
        tiff_ = TIFFOpen(filePath.c_str(), "r");
        if (tiff_ != nullptr) 
        {
            if (TIFFIsTiled(tiff_) == 0) 
            { 
                throw (std::runtime_error("Tile Loader ERROR: The file is not tiled.")); 
            }
            // Load/parse header
            uint32_t temp;  // Using this variable to correctly read 'uint32_t' TIFF field values into 'size_t' variables
            uint16_t compression;
            TIFFGetField(tiff_, TIFFTAG_COMPRESSION, &compression);
            TIFFGetField(tiff_, TIFFTAG_IMAGEWIDTH, &temp);
            this->fullWidth_ = temp;
            TIFFGetField(tiff_, TIFFTAG_IMAGELENGTH, &temp);
            this->fullHeight_ = temp;
            TIFFGetField(tiff_, TIFFTAG_TILEWIDTH, &temp);
            this->tileWidth_ = temp;
            TIFFGetField(tiff_, TIFFTAG_TILELENGTH, &temp);
            this->tileHeight_ = temp;
            TIFFGetField(tiff_, TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel);
            TIFFGetField(tiff_, TIFFTAG_BITSPERSAMPLE, &(this->bitsPerSample_));
            TIFFGetField(tiff_, TIFFTAG_SAMPLEFORMAT, &(this->sampleFormat_));

            // Test if the file is greyscale
            if (samplesPerPixel != 1) 
            {
                std::stringstream message;
                message << "Tile Loader ERROR: The file is not greyscale: SamplesPerPixel = " << samplesPerPixel << ".";
                throw (std::runtime_error(message.str()));
            }
            // Interpret undefined data format as unsigned integer data
            if (sampleFormat_ < 1 || sampleFormat_ > 3) 
            { 
                sampleFormat_ = 1; 
            }
        }
        else 
        { 
            throw (std::runtime_error("Tile Loader ERROR: The file can not be opened.")); 
        }
    }

    /// @brief NyxusGrayscaleTiffTileLoader destructor
    ~NyxusGrayscaleTiffTileLoader() override 
    {
        if (tiff_) 
        {
            TIFFClose(tiff_);
            tiff_ = nullptr;
        }
    }

    /// @brief Load a tiff tile from a view
    /// @param tile Tile to copy into
    /// @param indexRowGlobalTile Tile row index
    /// @param indexColGlobalTile Tile column index
     /// @param indexLayerGlobalTile Tile layer index
    /// @param level Tile's level
    void loadTileFromFile(std::shared_ptr<std::vector<DataType>> tile,
        size_t indexRowGlobalTile,
        size_t indexColGlobalTile,
        size_t indexLayerGlobalTile,
        size_t level) override
    {
        // Get ahold of the logical (feature extraction facing) tile buffer from its smart pointer
        std::vector<DataType>& tileDataVec = *tile;

        tdata_t tiffTile = nullptr;
        auto t_szb = TIFFTileSize(tiff_);
        tiffTile = _TIFFmalloc(t_szb);
        auto errcode = TIFFReadTile(tiff_, tiffTile, indexColGlobalTile * tileWidth_, indexRowGlobalTile * tileHeight_, 0, 0);
        if (errcode < 0)
        {
            std::stringstream message;
            message
                << "Tile Loader ERROR: error reading tile data returning code "
                << errcode;
            throw (std::runtime_error(message.str()));
        }
        std::stringstream message;
        switch (sampleFormat_) 
        {
        case 1:
            switch (bitsPerSample_) 
            {
            case 8:
                loadTile <uint8_t> (tiffTile, tileDataVec);
                break;
            case 16:
                loadTile <uint16_t> (tiffTile, tileDataVec);    
                break;
            case 32:
                loadTile <uint32_t> (tiffTile, tileDataVec);
                break;
            case 64:
                loadTile <uint64_t> (tiffTile, tileDataVec);
                break;
            default:
                message
                    << "Tile Loader ERROR: The data format is not supported for unsigned integer, number bits per pixel = "
                    << bitsPerSample_;
                throw (std::runtime_error(message.str()));
            }
            break;
        case 2:
            switch (bitsPerSample_) 
            {
            case 8:
                loadTile<int8_t>(tiffTile, tileDataVec);
                break;
            case 16:
                loadTile<int16_t>(tiffTile, tileDataVec);
                break;
            case 32:
                loadTile<int32_t>(tiffTile, tileDataVec);
                break;
            case 64:
                loadTile<int64_t>(tiffTile, tileDataVec);
                break;
            default:
                message
                    << "Tile Loader ERROR: The data format is not supported for signed integer, number bits per pixel = "
                    << bitsPerSample_;
                throw (std::runtime_error(message.str()));
            }
            break;
        case 3:
            switch (bitsPerSample_) 
            {
            case 8:
            case 16:
            case 32:
                loadTile<float>(tiffTile, tileDataVec);
                break;
            case 64:
                loadTile<double>(tiffTile, tileDataVec);
                break;
            default:
                message
                    << "Tile Loader ERROR: The data format is not supported for float, number bits per pixel = "
                    << bitsPerSample_;
                throw (std::runtime_error(message.str()));
            }
            break;
        default:
            message << "Tile Loader ERROR: The data format is not supported, sample format = " << sampleFormat_;
            throw (std::runtime_error(message.str()));
        }

        _TIFFfree(tiffTile);
    }


    /// @brief Tiff file height
    /// @param level Tiff level [not used]
    /// @return Full height
    [[nodiscard]] size_t fullHeight([[maybe_unused]] size_t level) const override { return fullHeight_; }
    /// @brief Tiff full width
    /// @param level Tiff level [not used]
    /// @return Full width
    [[nodiscard]] size_t fullWidth([[maybe_unused]] size_t level) const override { return fullWidth_; }
    /// @brief Tiff tile width
    /// @param level Tiff level [not used]
    /// @return Tile width
    [[nodiscard]] size_t tileWidth([[maybe_unused]] size_t level) const override { return tileWidth_; }
    /// @brief Tiff tile height
    /// @param level Tiff level [not used]
    /// @return Tile height
    [[nodiscard]] size_t tileHeight([[maybe_unused]] size_t level) const override { return tileHeight_; }
    /// @brief Tiff bits per sample
    /// @return Size of a sample in bits
    [[nodiscard]] short bitsPerSample() const override { return bitsPerSample_; }
    /// @brief Level accessor
    /// @return 1
    [[nodiscard]] size_t numberPyramidLevels() const override { return 1; }

private:

    #if 0   // A faster implementation is available. Keeping this for records.
    /// @brief Private function to copy and cast the values
    /// @tparam FileType Type inside the file
    /// @param src Piece of memory coming from libtiff
    /// @param dest Feature extraction facing logical buffer to fill
    /// 
    template<typename FileType>
    void loadTile(tdata_t src, std::shared_ptr<std::vector<DataType>>& dest)
    {
        for (size_t i = 0; i < tileHeight_ * tileWidth_; ++i)
        {
            // Logic to prevent "noise" in images whose dimensions are smaller than the default tile buffer size 1024x1024
            auto row = i / tileWidth_,
                col = i % tileHeight_;
            if (col < fullWidth_ && row < fullHeight_)
                dest->data()[i] = (DataType)((FileType*)(src))[i];
            else
                dest->data()[i] = (DataType)0;  // Zero-fill gaps
        }
    }
    #endif

    /// @brief Private function to copy and cast the values
    /// @tparam FileType Type inside the file
    /// @param src Piece of memory coming from libtiff
    /// @param dst_as_vector Feature extraction facing logical buffer to fill
    /// 
    template<typename FileType>
    void loadTile(tdata_t src, std::vector<DataType>& dst_as_vector)
    {
        // Get ahold of the raw pointer
        DataType* dest = dst_as_vector.data();

        // Special case of tileWidth_ (e.g. 1024) > fullWidth_ (e.g. 256)
        if (tileWidth_ > fullWidth_ && tileHeight_ > fullHeight_)
        {
            // Zero-prefill margins of the logical buffer 
            size_t szb = tileHeight_ * tileWidth_ * sizeof(*dest); 
            memset(dest, 0, szb);

            // Copy pixels assuming the row-major layout both in the physical (TIFF) and logical (ROI scanner facing) buffers
            for (size_t r = 0; r < fullHeight_; r++)
                for (size_t c = 0; c < fullWidth_; c++)
                {
                    size_t logOffs = r * tileWidth_ + c,
                        physOffs = r * tileWidth_ + c;
                    *(dest + logOffs) = (DataType) *(((FileType*)src) + physOffs);
                }
        }
        else
            // General case the logical buffer is same size (specifically, tile size) as the physical one even if tileWidth_ (e.g. 1024) < fullWidth_ (e.g. 1080)
            {
                size_t n = tileHeight_ * tileWidth_;
                for (size_t i = 0; i < n; i++)
                    *(dest + i) = (DataType) *(((FileType*)src) + i);
            }

    }

    TIFF*
        tiff_ = nullptr;             ///< Tiff file pointer

    size_t
        fullHeight_ = 0,           ///< Full height in pixel
        fullWidth_ = 0,            ///< Full width in pixel
        tileHeight_ = 0,            ///< Tile height
        tileWidth_ = 0;             ///< Tile width

    short
        sampleFormat_ = 0,          ///< Sample format as defined by libtiff
        bitsPerSample_ = 0;         ///< Bit Per Sample as defined by libtiff

};