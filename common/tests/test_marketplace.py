import pytest

from st_common.marketplace import SellerExportSource, SourceBlocked, get_source


@pytest.mark.parametrize("name", ["torob", "digikala", "basalam"])
async def test_marketplace_adapters_are_blocked_and_make_no_requests(name):
    src = get_source(name)
    assert src.access_status == "blocked_tos_review"
    with pytest.raises(SourceBlocked):
        await src.get_product("1")
    with pytest.raises(SourceBlocked):
        await src.get_categories()
    with pytest.raises(SourceBlocked):
        async for _ in src.iter_products():
            pass


def test_unknown_source():
    with pytest.raises(ValueError):
        get_source("amazon")


async def test_seller_export_source_normalizes(tmp_path):
    csv_path = tmp_path / "export.csv"
    csv_path.write_text(
        "product_id,title,category,price,attributes_json,images\n"
        '1,زعفران سرگل يك مثقالي,خوراکی > ادویه,۲۵۰ هزار تومان,"{""وزن"": ""٤٫٦ گرم""}",a.jpg|b.jpg\n'
        '2,فرش دستباف,فرش > دستباف,"12,000,000 ریال",{},\n',
        encoding="utf-8",
    )
    src = SellerExportSource(csv_path, consent="bot-optin-2026-09")
    items = [p async for p in src.iter_products()]
    assert items[0].title_fa == "زعفران سرگل یک مثقالی"
    assert items[0].price_toman == 250_000
    assert items[0].attributes == {"وزن": "۴٫۶ گرم"}
    assert items[0].image_refs == ["a.jpg", "b.jpg"]
    assert items[0].consent == "bot-optin-2026-09"
    assert [c.title_fa for c in await src.get_categories()] == ["خوراکی", "ادویه", "فرش", "دستباف"]
    assert (await src.get_product("1")).product_id == "1"
    assert [p.product_id async for p in src.iter_products(category_id="فرش")] == ["2"]
    with pytest.raises(KeyError):
        await src.get_product("404")
